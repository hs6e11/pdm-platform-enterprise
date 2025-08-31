#!/usr/bin/env python3
"""
Production Industrial Protocol Clients
Real Modbus, OPC-UA, MQTT integration replacing simulated data
"""

import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json
import struct
from dataclasses import dataclass, asdict
import ssl

# Protocol-specific imports
from pymodbus.client.asynchronous.tcp import AsyncModbusTCPClient
from pymodbus.client.asynchronous.serial import AsyncModbusSerialClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian

try:
    from opcua import Client as OPCClient, ua
    from opcua.common.node import Node
except ImportError:
    logger.warning("OPC-UA library not installed. Install with: pip install opcua")
    OPCClient = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    logger.warning("MQTT library not installed. Install with: pip install paho-mqtt")
    mqtt = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    equipment_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime
    quality: str = "good"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EquipmentConfig:
    equipment_id: str
    protocol: str  # modbus, opcua, mqtt
    connection_params: Dict[str, Any]
    sensor_mappings: Dict[str, Dict[str, Any]]
    tenant_id: str
    polling_interval: int = 30  # seconds
    enabled: bool = True

class BaseProtocolClient(ABC):
    """Base class for all protocol clients"""
    
    def __init__(self, equipment_config: EquipmentConfig):
        self.config = equipment_config
        self.connected = False
        self.last_reading_time = None
        self.error_count = 0
        self.max_errors = 5
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to device"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to device"""
        pass
    
    @abstractmethod
    async def read_sensors(self) -> List[SensorReading]:
        """Read all configured sensors"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Return client health status"""
        return {
            'equipment_id': self.config.equipment_id,
            'protocol': self.config.protocol,
            'connected': self.connected,
            'last_reading': self.last_reading_time.isoformat() if self.last_reading_time else None,
            'error_count': self.error_count,
            'status': 'healthy' if self.connected and self.error_count < self.max_errors else 'error'
        }

class ModbusClient(BaseProtocolClient):
    """Modbus TCP/RTU client for industrial equipment"""
    
    def __init__(self, equipment_config: EquipmentConfig):
        super().__init__(equipment_config)
        self.client = None
        self.slave_id = equipment_config.connection_params.get('slave_id', 1)
        
    async def connect(self) -> bool:
        """Connect to Modbus device"""
        try:
            params = self.config.connection_params
            
            if params.get('connection_type') == 'tcp':
                self.client = AsyncModbusTCPClient(
                    host=params['host'],
                    port=params.get('port', 502),
                    timeout=params.get('timeout', 10)
                )
            elif params.get('connection_type') == 'serial':
                self.client = AsyncModbusSerialClient(
                    method='rtu',
                    port=params['port'],
                    baudrate=params.get('baudrate', 9600),
                    bytesize=params.get('bytesize', 8),
                    parity=params.get('parity', 'N'),
                    stopbits=params.get('stopbits', 1),
                    timeout=params.get('timeout', 10)
                )
            else:
                raise ValueError("Invalid Modbus connection type")
            
            # Start the client
            await self.client.start()
            self.connected = True
            self.error_count = 0
            
            logger.info(f"Connected to Modbus device: {self.config.equipment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Modbus {self.config.equipment_id}: {str(e)}")
            self.connected = False
            self.error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from Modbus device"""
        try:
            if self.client:
                await self.client.stop()
                self.connected = False
                logger.info(f"Disconnected from Modbus device: {self.config.equipment_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from Modbus {self.config.equipment_id}: {str(e)}")
    
    def _decode_register_value(self, registers: List[int], data_type: str, byte_order: str = 'big') -> float:
        """Decode Modbus register values based on data type"""
        try:
            if not registers:
                return 0.0
            
            if data_type in ['int16', 'uint16']:
                return float(registers[0])
            
            elif data_type in ['int32', 'uint32', 'float32']:
                if len(registers) < 2:
                    return 0.0
                
                # Combine registers into 32-bit value
                if byte_order == 'big':
                    value_bytes = struct.pack('>HH', registers[0], registers[1])
                else:
                    value_bytes = struct.pack('<HH', registers[1], registers[0])
                
                if data_type == 'int32':
                    return float(struct.unpack('>i', value_bytes)[0])
                elif data_type == 'uint32':
                    return float(struct.unpack('>I', value_bytes)[0])
                elif data_type == 'float32':
                    return struct.unpack('>f', value_bytes)[0]
            
            elif data_type == 'float64':
                if len(registers) < 4:
                    return 0.0
                
                if byte_order == 'big':
                    value_bytes = struct.pack('>HHHH', *registers[:4])
                else:
                    value_bytes = struct.pack('<HHHH', *reversed(registers[:4]))
                
                return struct.unpack('>d', value_bytes)[0]
            
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return float(registers[0])
                
        except Exception as e:
            logger.error(f"Error decoding register value: {str(e)}")
            return 0.0
    
    async def read_sensors(self) -> List[SensorReading]:
        """Read all configured sensors from Modbus device"""
        readings = []
        timestamp = datetime.now(timezone.utc)
        
        if not self.connected:
            if not await self.connect():
                return readings
        
        try:
            for sensor_type, mapping in self.config.sensor_mappings.items():
                try:
                    register_type = mapping.get('register_type', 'holding')
                    address = mapping['address']
                    count = mapping.get('count', 1)
                    data_type = mapping.get('data_type', 'uint16')
                    scale_factor = mapping.get('scale_factor', 1.0)
                    offset = mapping.get('offset', 0.0)
                    unit = mapping.get('unit', 'unknown')
                    
                    # Read registers based on type
                    if register_type == 'holding':
                        response = await self.client.read_holding_registers(
                            address, count, unit=self.slave_id
                        )
                    elif register_type == 'input':
                        response = await self.client.read_input_registers(
                            address, count, unit=self.slave_id
                        )
                    elif register_type == 'coil':
                        response = await self.client.read_coils(
                            address, count, unit=self.slave_id
                        )
                        # Convert boolean to float
                        if not response.isError():
                            value = float(response.bits[0]) * scale_factor + offset
                            readings.append(SensorReading(
                                equipment_id=self.config.equipment_id,
                                sensor_type=sensor_type,
                                value=value,
                                unit=unit,
                                timestamp=timestamp,
                                metadata={
                                    'protocol': 'modbus',
                                    'address': address,
                                    'register_type': register_type
                                }
                            ))
                        continue
                    
                    elif register_type == 'discrete':
                        response = await self.client.read_discrete_inputs(
                            address, count, unit=self.slave_id
                        )
                        # Convert boolean to float
                        if not response.isError():
                            value = float(response.bits[0]) * scale_factor + offset
                            readings.append(SensorReading(
                                equipment_id=self.config.equipment_id,
                                sensor_type=sensor_type,
                                value=value,
                                unit=unit,
                                timestamp=timestamp,
                                metadata={
                                    'protocol': 'modbus',
                                    'address': address,
                                    'register_type': register_type
                                }
                            ))
                        continue
                    
                    # Handle register responses
                    if response.isError():
                        logger.error(f"Modbus read error for {sensor_type}: {response}")
                        continue
                    
                    # Decode value
                    raw_value = self._decode_register_value(
                        response.registers, data_type, mapping.get('byte_order', 'big')
                    )
                    
                    # Apply scaling and offset
                    value = raw_value * scale_factor + offset
                    
                    readings.append(SensorReading(
                        equipment_id=self.config.equipment_id,
                        sensor_type=sensor_type,
                        value=value,
                        unit=unit,
                        timestamp=timestamp,
                        metadata={
                            'protocol': 'modbus',
                            'address': address,
                            'register_type': register_type,
                            'raw_value': raw_value
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Error reading {sensor_type} from {self.config.equipment_id}: {str(e)}")
            
            self.last_reading_time = timestamp
            self.error_count = 0
            
        except Exception as e:
            logger.error(f"Critical error reading from {self.config.equipment_id}: {str(e)}")
            self.error_count += 1
            self.connected = False
        
        logger.info(f"Read {len(readings)} sensors from Modbus device {self.config.equipment_id}")
        return readings

class OPCUAClient(BaseProtocolClient):
    """OPC-UA client for industrial equipment"""
    
    def __init__(self, equipment_config: EquipmentConfig):
        super().__init__(equipment_config)
        self.client = None
        
    async def connect(self) -> bool:
        """Connect to OPC-UA server"""
        if OPCClient is None:
            logger.error("OPC-UA library not available")
            return False
        
        try:
            params = self.config.connection_params
            endpoint = params['endpoint']
            
            self.client = OPCClient(endpoint)
            
            # Configure security
            if params.get('security_mode'):
                self.client.set_security_string(params['security_mode'])
            
            # Authentication
            if params.get('username') and params.get('password'):
                self.client.set_user(params['username'])
                self.client.set_password(params['password'])
            
            # Connect
            await asyncio.get_event_loop().run_in_executor(None, self.client.connect)
            
            self.connected = True
            self.error_count = 0
            
            logger.info(f"Connected to OPC-UA server: {self.config.equipment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OPC-UA {self.config.equipment_id}: {str(e)}")
            self.connected = False
            self.error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from OPC-UA server"""
        try:
            if self.client:
                await asyncio.get_event_loop().run_in_executor(None, self.client.disconnect)
                self.connected = False
                logger.info(f"Disconnected from OPC-UA server: {self.config.equipment_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from OPC-UA {self.config.equipment_id}: {str(e)}")
    
    async def read_sensors(self) -> List[SensorReading]:
        """Read all configured sensors from OPC-UA server"""
        readings = []
        timestamp = datetime.now(timezone.utc)
        
        if not self.connected:
            if not await self.connect():
                return readings
        
        try:
            for sensor_type, mapping in self.config.sensor_mappings.items():
                try:
                    node_id = mapping['node_id']
                    unit = mapping.get('unit', 'unknown')
                    scale_factor = mapping.get('scale_factor', 1.0)
                    offset = mapping.get('offset', 0.0)
                    
                    # Get node and read value
                    node = self.client.get_node(node_id)
                    value = await asyncio.get_event_loop().run_in_executor(None, node.get_value)
                    
                    # Convert to float and apply scaling
                    if isinstance(value, (int, float)):
                        scaled_value = float(value) * scale_factor + offset
                    elif isinstance(value, bool):
                        scaled_value = float(value) * scale_factor + offset
                    else:
                        logger.warning(f"Unexpected value type for {sensor_type}: {type(value)}")
                        scaled_value = 0.0
                    
                    readings.append(SensorReading(
                        equipment_id=self.config.equipment_id,
                        sensor_type=sensor_type,
                        value=scaled_value,
                        unit=unit,
                        timestamp=timestamp,
                        metadata={
                            'protocol': 'opcua',
                            'node_id': node_id,
                            'raw_value': value
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Error reading {sensor_type} from OPC-UA {self.config.equipment_id}: {str(e)}")
            
            self.last_reading_time = timestamp
            self.error_count = 0
            
        except Exception as e:
            logger.error(f"Critical error reading from OPC-UA {self.config.equipment_id}: {str(e)}")
            self.error_count += 1
            self.connected = False
        
        logger.info(f"Read {len(readings)} sensors from OPC-UA server {self.config.equipment_id}")
        return readings

class MQTTClient(BaseProtocolClient):
    """MQTT client for industrial equipment"""
    
    def __init__(self, equipment_config: EquipmentConfig):
        super().__init__(equipment_config)
        self.client = None
        self.received_data = {}
        self.data_lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        if mqtt is None:
            logger.error("MQTT library not available")
            return False
        
        try:
            params = self.config.connection_params
            
            self.client = mqtt.Client()
            
            # Configure authentication
            if params.get('username') and params.get('password'):
                self.client.username_pw_set(params['username'], params['password'])
            
            # Configure SSL/TLS
            if params.get('use_tls'):
                context = ssl.create_default_context()
                self.client.tls_set_context(context)
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Connect
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.client.connect,
                params['host'],
                params.get('port', 1883),
                params.get('keepalive', 60)
            )
            
            # Start loop
            self.client.loop_start()
            
            # Subscribe to topics
            for sensor_type, mapping in self.config.sensor_mappings.items():
                topic = mapping['topic']
                qos = mapping.get('qos', 0)
                self.client.subscribe(topic, qos)
                logger.info(f"Subscribed to MQTT topic: {topic}")
            
            self.connected = True
            self.error_count = 0
            
            logger.info(f"Connected to MQTT broker: {self.config.equipment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT {self.config.equipment_id}: {str(e)}")
            self.connected = False
            self.error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logger.info(f"Disconnected from MQTT broker: {self.config.equipment_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT {self.config.equipment_id}: {str(e)}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT connected with result code {rc}")
        else:
            logger.error(f"MQTT connection failed with result code {rc}")
            self.connected = False
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            # Store received data with timestamp
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            timestamp = datetime.now(timezone.utc)
            
            # Try to parse as JSON
            try:
                data = json.loads(payload)
                if isinstance(data, (int, float)):
                    value = float(data)
                elif isinstance(data, dict) and 'value' in data:
                    value = float(data['value'])
                else:
                    logger.warning(f"Unexpected MQTT payload format: {payload}")
                    return
            except json.JSONDecodeError:
                # Try to parse as plain number
                try:
                    value = float(payload)
                except ValueError:
                    logger.warning(f"Cannot parse MQTT payload: {payload}")
                    return
            
            # Store with lock
            asyncio.create_task(self._store_mqtt_data(topic, value, timestamp))
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {str(e)}")
    
    async def _store_mqtt_data(self, topic: str, value: float, timestamp: datetime):
        """Store MQTT data with async lock"""
        async with self.data_lock:
            self.received_data[topic] = {
                'value': value,
                'timestamp': timestamp
            }
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection: {rc}")
    
    async def read_sensors(self) -> List[SensorReading]:
        """Read all sensors from cached MQTT data"""
        readings = []
        current_time = datetime.now(timezone.utc)
        
        if not self.connected:
            if not await self.connect():
                return readings
        
        async with self.data_lock:
            for sensor_type, mapping in self.config.sensor_mappings.items():
                topic = mapping['topic']
                unit = mapping.get('unit', 'unknown')
                scale_factor = mapping.get('scale_factor', 1.0)
                offset = mapping.get('offset', 0.0)
                max_age = mapping.get('max_age_seconds', 300)  # 5 minutes default
                
                if topic in self.received_data:
                    data = self.received_data[topic]
                    data_age = (current_time - data['timestamp']).total_seconds()
                    
                    if data_age <= max_age:
                        # Apply scaling and offset
                        scaled_value = data['value'] * scale_factor + offset
                        
                        readings.append(SensorReading(
                            equipment_id=self.config.equipment_id,
                            sensor_type=sensor_type,
                            value=scaled_value,
                            unit=unit,
                            timestamp=data['timestamp'],
                            quality="good" if data_age <= 60 else "stale",
                            metadata={
                                'protocol': 'mqtt',
                                'topic': topic,
                                'raw_value': data['value'],
                                'age_seconds': data_age
                            }
                        ))
                    else:
                        logger.warning(f"MQTT data for {sensor_type} is too old: {data_age}s")
                else:
                    logger.warning(f"No MQTT data received for {sensor_type} (topic: {topic})")
        
        if readings:
            self.last_reading_time = current_time
            self.error_count = 0
        
        logger.info(f"Read {len(readings)} sensors from MQTT {self.config.equipment_id}")
        return readings

class ProtocolManager:
    """Manages multiple protocol clients and data collection"""
    
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint
        self.clients: Dict[str, BaseProtocolClient] = {}
        self.running = False
        
    def add_equipment(self, equipment_config: EquipmentConfig):
        """Add equipment with protocol client"""
        if equipment_config.protocol.lower() == 'modbus':
            client = ModbusClient(equipment_config)
        elif equipment_config.protocol.lower() == 'opcua':
            client = OPCUAClient(equipment_config)
        elif equipment_config.protocol.lower() == 'mqtt':
            client = MQTTClient(equipment_config)
        else:
            raise ValueError(f"Unsupported protocol: {equipment_config.protocol}")
        
        self.clients[equipment_config.equipment_id] = client
        logger.info(f"Added {equipment_config.protocol.upper()} client for {equipment_config.equipment_id}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all clients"""
        results = {}
        
        for equipment_id, client in self.clients.items():
            if client.config.enabled:
                results[equipment_id] = await client.connect()
            else:
                results[equipment_id] = False
                logger.info(f"Equipment {equipment_id} is disabled")
        
        connected_count = sum(results.values())
        logger.info(f"Connected {connected_count}/{len(results)} equipment")
        return results
    
    async def disconnect_all(self):
        """Disconnect all clients"""
        for client in self.clients.values():
            await client.disconnect()
        logger.info("All clients disconnected")
    
    async def collect_all_data(self) -> List[SensorReading]:
        """Collect data from all connected clients"""
        all_readings = []
        
        for equipment_id, client in self.clients.items():
            if client.config.enabled and client.connected:
                try:
                    readings = await client.read_sensors()
                    all_readings.extend(readings)
                except Exception as e:
                    logger.error(f"Error collecting data from {equipment_id}: {str(e)}")
        
        logger.info(f"Collected {len(all_readings)} total sensor readings")
        return all_readings
    
    async def send_to_api(self, readings: List[SensorReading]) -> bool:
        """Send readings to PDM API"""
        if not readings:
            return True
        
        try:
            # Group readings by tenant for multi-tenant API
            tenant_groups = {}
            for reading in readings:
                # Find tenant for equipment (simplified - you might have a mapping)
                equipment_client = self.clients.get(reading.equipment_id)
                if equipment_client:
                    tenant_id = equipment_client.config.tenant_id
                    if tenant_id not in tenant_groups:
                        tenant_groups[tenant_id] = []
                    tenant_groups[tenant_id].append(asdict(reading))
            
            # Send to API endpoint
            async with aiohttp.ClientSession() as session:
                for tenant_id, tenant_readings in tenant_groups.items():
                    payload = {
                        'tenant_id': tenant_id,
                        'readings': tenant_readings
                    }
                    
                    async with session.post(
                        f"{self.api_endpoint}/api/readings/bulk",
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Sent {len(tenant_readings)} readings for tenant {tenant_id}")
                        else:
                            logger.error(f"Failed to send readings: {response.status}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to API: {str(e)}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all clients"""
        health_data = {
            'overall_status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equipment_status': {},
            'summary': {
                'total_equipment': len(self.clients),
                'connected': 0,
                'errors': 0
            }
        }
        
        for equipment_id, client in self.clients.items():
            client_health = await client.health_check()
            health_data['equipment_status'][equipment_id] = client_health
            
            if client_health['connected']:
                health_data['summary']['connected'] += 1
            if client_health['status'] == 'error':
                health_data['summary']['errors'] += 1
        
        # Overall health assessment
        if health_data['summary']['errors'] > 0:
            health_data['overall_status'] = 'degraded'
        if health_data['summary']['connected'] == 0:
            health_data['overall_status'] = 'critical'
        
        return health_data
    
    async def run_continuous_collection(self, interval: int = 30):
        """Run continuous data collection loop"""
        logger.info(f"Starting continuous data collection with {interval}s interval")
        self.running = True
        
        # Connect all clients
        await self.connect_all()
        
        try:
            while self.running:
                # Collect data
                readings = await self.collect_all_data()
                
                # Send to API
                if readings:
                    await self.send_to_api(readings)
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping continuous collection...")
        except Exception as e:
            logger.error(f"Error in continuous collection: {str(e)}")
        finally:
            self.running = False
            await self.disconnect_all()

def load_equipment_config(config_file: str = "equipment_config.json") -> List[EquipmentConfig]:
    """Load equipment configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        equipment_configs = []
        for item in config_data['equipment']:
            config = EquipmentConfig(**item)
            equipment_configs.append(config)
        
        logger.info(f"Loaded configuration for {len(equipment_configs)} equipment")
        return equipment_configs
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return []

async def main():
    """Main execution for protocol clients"""
    # Load configuration
    configs = load_equipment_config()
    
    if not configs:
        # Create sample configuration
        sample_config = [
            EquipmentConfig(
                equipment_id="EG_M001",
                protocol="modbus",
                connection_params={
                    "connection_type": "tcp",
                    "host": "192.168.1.100",
                    "port": 502,
                    "timeout": 10
                },
                sensor_mappings={
                    "vibration_x": {"address": 0, "data_type": "float32", "unit": "mm/s", "count": 2},
                    "vibration_y": {"address": 2, "data_type": "float32", "unit": "mm/s", "count": 2},
                    "temperature": {"address": 4, "data_type": "int16", "unit": "°C", "scale_factor": 0.1},
                    "pressure": {"address": 5, "data_type": "int16", "unit": "bar", "scale_factor": 0.01},
                    "speed_rpm": {"address": 6, "data_type": "uint16", "unit": "rpm"}
                },
                tenant_id="550e8400-e29b-41d4-a716-446655440000",
                polling_interval=30
            )
        ]
        
        # Save sample configuration
        with open("equipment_config.json", 'w') as f:
            json.dump({
                "equipment": [asdict(config) for config in sample_config]
            }, f, indent=2, default=str)
        
        logger.info("Created sample configuration file: equipment_config.json")
        configs = sample_config
    
    # Initialize protocol manager
    api_endpoint = os.getenv('PDM_API_ENDPOINT', 'http://localhost:3000')
    manager = ProtocolManager(api_endpoint)
    
    # Add all equipment
    for config in configs:
        manager.add_equipment(config)
    
    # Start continuous collection
    await manager.run_continuous_collection(interval=30)

if __name__ == "__main__":
    asyncio.run(main())
