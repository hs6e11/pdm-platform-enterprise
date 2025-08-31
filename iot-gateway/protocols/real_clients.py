# iot-gateway/protocols/real_clients.py
"""
Real industrial protocol implementations for PDM Platform v2.0
Replaces simulated data with actual industrial protocol integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import json
import uuid

# Industrial protocol libraries
from pymodbus.client import ModbusTcpClient, ModbusSerialClient
from pymodbus.exceptions import ModbusException
import paho.mqtt.client as mqtt_client
from opcua import Client as OPCUAClient, ua
from opcua.common.node import Node
import aiohttp
import ssl

logger = logging.getLogger(__name__)

class ProtocolError(Exception):
    """Custom exception for protocol communication errors"""
    pass

class BaseProtocolClient(ABC):
    """Abstract base class for all protocol clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client_id = config.get('client_id', str(uuid.uuid4()))
        self.is_connected = False
        self.last_error = None
        self.connection_attempts = 0
        self.max_retries = config.get('max_retries', 3)
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the protocol server/device"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to the protocol server/device"""
        pass
    
    @abstractmethod
    async def read_sensors(self) -> List[Dict[str, Any]]:
        """Read sensor data and return standardized format"""
        pass
    
    def _standardize_reading(self, machine_id: str, sensor_type: str, value: float, 
                           timestamp: datetime = None, metadata: Dict = None) -> Dict[str, Any]:
        """Standardize sensor reading format across all protocols"""
        return {
            'machine_id': machine_id,
            'sensor_type': sensor_type,
            'value': float(value),
            'timestamp': timestamp or datetime.utcnow(),
            'protocol': self.__class__.__name__.replace('Client', '').lower(),
            'client_id': self.client_id,
            'metadata': metadata or {}
        }

class ModbusClient(BaseProtocolClient):
    """Real Modbus TCP/RTU client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config['host']
        self.port = config.get('port', 502)
        self.unit_id = config.get('unit_id', 1)
        self.protocol_type = config.get('type', 'tcp')  # 'tcp' or 'rtu'
        self.registers = config.get('registers', {})
        self.client = None
        
    async def connect(self) -> bool:
        """Connect to Modbus device"""
        try:
            if self.protocol_type.lower() == 'tcp':
                self.client = ModbusTcpClient(self.host, port=self.port)
            else:
                # RTU over serial
                self.client = ModbusSerialClient(
                    port=self.host,  # Serial port path
                    baudrate=self.config.get('baudrate', 9600),
                    parity=self.config.get('parity', 'N'),
                    stopbits=self.config.get('stopbits', 1),
                    timeout=self.config.get('timeout', 3)
                )
            
            connection_result = self.client.connect()
            self.is_connected = connection_result
            
            if connection_result:
                logger.info(f"Connected to Modbus device at {self.host}:{self.port}")
                self.connection_attempts = 0
            else:
                self.connection_attempts += 1
                logger.error(f"Failed to connect to Modbus device (attempt {self.connection_attempts})")
            
            return connection_result
            
        except Exception as e:
            self.last_error = str(e)
            self.connection_attempts += 1
            logger.error(f"Modbus connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Modbus device"""
        if self.client:
            try:
                self.client.close()
                self.is_connected = False
                logger.info("Modbus client disconnected")
            except Exception as e:
                logger.error(f"Modbus disconnect error: {e}")
    
    async def read_sensors(self) -> List[Dict[str, Any]]:
        """Read sensor data from Modbus registers"""
        if not self.is_connected:
            raise ProtocolError("Modbus client not connected")
        
        readings = []
        timestamp = datetime.utcnow()
        
        try:
            for machine_id, machine_config in self.registers.items():
                for sensor_config in machine_config.get('sensors', []):
                    register_address = sensor_config['address']
                    sensor_type = sensor_config['type']
                    register_type = sensor_config.get('register_type', 'holding')
                    count = sensor_config.get('count', 1)
                    scale_factor = sensor_config.get('scale_factor', 1.0)
                    
                    # Read based on register type
                    if register_type == 'holding':
                        result = self.client.read_holding_registers(register_address, count, slave=self.unit_id)
                    elif register_type == 'input':
                        result = self.client.read_input_registers(register_address, count, slave=self.unit_id)
                    elif register_type == 'coil':
                        result = self.client.read_coils(register_address, count, slave=self.unit_id)
                    elif register_type == 'discrete':
                        result = self.client.read_discrete_inputs(register_address, count, slave=self.unit_id)
                    else:
                        logger.warning(f"Unknown register type: {register_type}")
                        continue
                    
                    if result.isError():
                        logger.error(f"Modbus read error for {machine_id}.{sensor_type}: {result}")
                        continue
                    
                    # Process the result
                    if register_type in ['holding', 'input']:
                        value = result.registers[0] * scale_factor
                        if count > 1:
                            # Handle multi-register values (e.g., 32-bit floats)
                            value = self._combine_registers(result.registers, sensor_config.get('data_type', 'uint16'))
                    else:
                        value = float(result.bits[0])
                    
                    reading = self._standardize_reading(
                        machine_id=machine_id,
                        sensor_type=sensor_type,
                        value=value,
                        timestamp=timestamp,
                        metadata={
                            'register_address': register_address,
                            'register_type': register_type,
                            'unit_id': self.unit_id
                        }
                    )
                    readings.append(reading)
            
            logger.info(f"Read {len(readings)} Modbus sensor values")
            return readings
            
        except ModbusException as e:
            logger.error(f"Modbus protocol error: {e}")
            self.is_connected = False
            raise ProtocolError(f"Modbus read failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected Modbus error: {e}")
            raise ProtocolError(f"Modbus read failed: {e}")
    
    def _combine_registers(self, registers: List[int], data_type: str) -> float:
        """Combine multiple registers into a single value"""
        if data_type == 'float32':
            # IEEE 754 32-bit float from two 16-bit registers
            import struct
            bytes_data = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>f', bytes_data)[0]
        elif data_type == 'uint32':
            return (registers[0] << 16) | registers[1]
        else:
            return float(registers[0])

class OPCUAClient(BaseProtocolClient):
    """Real OPC-UA client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint_url = config['endpoint_url']
        self.security_policy = config.get('security_policy', 'Basic256Sha256')
        self.security_mode = config.get('security_mode', 'SignAndEncrypt')
        self.username = config.get('username')
        self.password = config.get('password')
        self.certificate_path = config.get('certificate_path')
        self.private_key_path = config.get('private_key_path')
        self.nodes = config.get('nodes', {})
        self.client = None
        
    async def connect(self) -> bool:
        """Connect to OPC-UA server"""
        try:
            self.client = OPCUAClient(self.endpoint_url)
            
            # Configure security if specified
            if self.certificate_path and self.private_key_path:
                self.client.set_security(
                    getattr(ua.SecurityPolicy, self.security_policy),
                    self.certificate_path,
                    self.private_key_path,
                    None,
                    getattr(ua.MessageSecurityMode, self.security_mode)
                )
            
            # Set authentication if specified
            if self.username and self.password:
                self.client.set_user(self.username)
                self.client.set_password(self.password)
            
            await asyncio.get_event_loop().run_in_executor(None, self.client.connect)
            self.is_connected = True
            logger.info(f"Connected to OPC-UA server at {self.endpoint_url}")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.connection_attempts += 1
            logger.error(f"OPC-UA connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OPC-UA server"""
        if self.client:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self.client.disconnect)
                self.is_connected = False
                logger.info("OPC-UA client disconnected")
            except Exception as e:
                logger.error(f"OPC-UA disconnect error: {e}")
    
    async def read_sensors(self) -> List[Dict[str, Any]]:
        """Read sensor data from OPC-UA nodes"""
        if not self.is_connected:
            raise ProtocolError("OPC-UA client not connected")
        
        readings = []
        timestamp = datetime.utcnow()
        
        try:
            for machine_id, machine_config in self.nodes.items():
                for sensor_config in machine_config.get('sensors', []):
                    node_id = sensor_config['node_id']
                    sensor_type = sensor_config['type']
                    
                    # Get node and read value
                    node = self.client.get_node(node_id)
                    value = await asyncio.get_event_loop().run_in_executor(None, node.get_value)
                    
                    # Get additional node attributes
                    display_name = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: node.get_display_name().Text
                    )
                    
                    reading = self._standardize_reading(
                        machine_id=machine_id,
                        sensor_type=sensor_type,
                        value=float(value),
                        timestamp=timestamp,
                        metadata={
                            'node_id': node_id,
                            'display_name': display_name,
                            'data_type': str(type(value).__name__)
                        }
                    )
                    readings.append(reading)
            
            logger.info(f"Read {len(readings)} OPC-UA sensor values")
            return readings
            
        except Exception as e:
            logger.error(f"OPC-UA read error: {e}")
            self.is_connected = False
            raise ProtocolError(f"OPC-UA read failed: {e}")

class MQTTClient(BaseProtocolClient):
    """Real MQTT client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.broker_host = config['host']
        self.broker_port = config.get('port', 1883)
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_tls = config.get('tls', False)
        self.topics = config.get('topics', {})
        self.client = None
        self.received_messages = []
        
    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client = mqtt_client.Client(client_id=self.client_id)
            
            # Configure authentication
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Configure TLS if required
            if self.use_tls:
                self.client.tls_set(ca_certs=None, certfile=None, keyfile=None, 
                                   cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
            
            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Connect to broker
            result = self.client.connect(self.broker_host, self.broker_port, 60)
            if result == 0:
                self.client.loop_start()
                self.is_connected = True
                logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
                return True
            else:
                logger.error(f"MQTT connection failed with code: {result}")
                return False
                
        except Exception as e:
            self.last_error = str(e)
            self.connection_attempts += 1
            logger.error(f"MQTT connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                self.is_connected = False
                logger.info("MQTT client disconnected")
            except Exception as e:
                logger.error(f"MQTT disconnect error: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful MQTT connection"""
        if rc == 0:
            logger.info("MQTT client connected successfully")
            # Subscribe to configured topics
            for machine_id, machine_config in self.topics.items():
                for sensor_config in machine_config.get('sensors', []):
                    topic = sensor_config['topic']
                    qos = sensor_config.get('qos', 1)
                    client.subscribe(topic, qos)
                    logger.info(f"Subscribed to MQTT topic: {topic}")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback for received MQTT messages"""
        try:
            # Parse message payload
            payload = json.loads(msg.payload.decode())
            
            # Store message for processing
            message_data = {
                'topic': msg.topic,
                'payload': payload,
                'timestamp': datetime.utcnow(),
                'qos': msg.qos
            }
            self.received_messages.append(message_data)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in MQTT message from topic {msg.topic}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.is_connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (code: {rc})")
    
    async def read_sensors(self) -> List[Dict[str, Any]]:
        """Process received MQTT messages into sensor readings"""
        if not self.is_connected:
            raise ProtocolError("MQTT client not connected")
        
        readings = []
        
        # Process all received messages
        messages_to_process = self.received_messages.copy()
        self.received_messages.clear()
        
        for message in messages_to_process:
            try:
                # Find the sensor configuration for this topic
                sensor_config = None
                machine_id = None
                
                for mid, machine_config in self.topics.items():
                    for sensor in machine_config.get('sensors', []):
                        if sensor['topic'] == message['topic']:
                            sensor_config = sensor
                            machine_id = mid
                            break
                    if sensor_config:
                        break
                
                if not sensor_config:
                    logger.warning(f"No configuration found for topic: {message['topic']}")
                    continue
                
                # Extract value from payload
                payload = message['payload']
                value_path = sensor_config.get('value_path', 'value')
                
                # Navigate nested JSON structure
                value = payload
                for key in value_path.split('.'):
                    value = value[key]
                
                reading = self._standardize_reading(
                    machine_id=machine_id,
                    sensor_type=sensor_config['type'],
                    value=float(value),
                    timestamp=message['timestamp'],
                    metadata={
                        'topic': message['topic'],
                        'qos': message['qos'],
                        'raw_payload': payload
                    }
                )
                readings.append(reading)
                
            except Exception as e:
                logger.error(f"Error processing MQTT message: {e}")
                continue
        
        logger.info(f"Processed {len(readings)} MQTT sensor messages")
        return readings

class HTTPClient(BaseProtocolClient):
    """HTTP/REST client for web-enabled sensors"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config['base_url']
        self.endpoints = config.get('endpoints', {})
        self.auth_headers = config.get('auth_headers', {})
        self.timeout = config.get('timeout', 30)
        self.session = None
        
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.auth_headers
            )
            self.is_connected = True
            logger.info(f"HTTP client initialized for {self.base_url}")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"HTTP client initialization error: {e}")
            return False
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            try:
                await self.session.close()
                self.is_connected = False
                logger.info("HTTP client session closed")
            except Exception as e:
                logger.error(f"HTTP disconnect error: {e}")
    
    async def read_sensors(self) -> List[Dict[str, Any]]:
        """Read sensor data from HTTP endpoints"""
        if not self.is_connected:
            raise ProtocolError("HTTP client not connected")
        
        readings = []
        timestamp = datetime.utcnow()
        
        try:
            for machine_id, machine_config in self.endpoints.items():
                for sensor_config in machine_config.get('sensors', []):
                    endpoint = sensor_config['endpoint']
                    sensor_type = sensor_config['type']
                    method = sensor_config.get('method', 'GET')
                    
                    url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
                    
                    async with self.session.request(method, url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Extract value using configured path
                            value_path = sensor_config.get('value_path', 'value')
                            value = data
                            for key in value_path.split('.'):
                                value = value[key]
                            
                            reading = self._standardize_reading(
                                machine_id=machine_id,
                                sensor_type=sensor_type,
                                value=float(value),
                                timestamp=timestamp,
                                metadata={
                                    'endpoint': endpoint,
                                    'http_status': response.status,
                                    'response_data': data
                                }
                            )
                            readings.append(reading)
                        else:
                            logger.error(f"HTTP request failed for {url}: {response.status}")
            
            logger.info(f"Read {len(readings)} HTTP sensor values")
            return readings
            
        except Exception as e:
            logger.error(f"HTTP read error: {e}")
            raise ProtocolError(f"HTTP read failed: {e}")

# Protocol client factory
def create_protocol_client(protocol_type: str, config: Dict[str, Any]) -> BaseProtocolClient:
    """Factory function to create appropriate protocol client"""
    clients = {
        'modbus': ModbusClient,
        'opcua': OPCUAClient,
        'mqtt': MQTTClient,
        'http': HTTPClient
    }
    
    client_class = clients.get(protocol_type.lower())
    if not client_class:
        raise ValueError(f"Unsupported protocol type: {protocol_type}")
    
    return client_class(config)

# Export main components
__all__ = [
    'BaseProtocolClient',
    'ModbusClient', 
    'OPCUAClient',
    'MQTTClient',
    'HTTPClient',
    'ProtocolError',
    'create_protocol_client'
]
