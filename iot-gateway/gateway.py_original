"""
Multi-Protocol IoT Gateway v2.0
Supports Modbus, OPC-UA, MQTT, HTTP protocols
"""

import asyncio
import aiohttp
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    machine_id: str
    sensor_type: str
    value: float
    timestamp: str
    quality: str = "GOOD"

class ProtocolClient(ABC):
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def read_sensors(self, config: Dict) -> List[SensorReading]:
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass

class ModbusClient(ProtocolClient):
    """Basic Modbus TCP client"""
    
    def __init__(self, host: str, port: int = 502):
        self.host = host
        self.port = port
        self.connected = False
    
    async def connect(self) -> bool:
        # Simulate connection for now
        logger.info(f"Connecting to Modbus {self.host}:{self.port}")
        await asyncio.sleep(1)
        self.connected = True
        return True
    
    async def read_sensors(self, config: Dict) -> List[SensorReading]:
        readings = []
        if not self.connected:
            return readings
        
        # Simulate reading sensors
        for sensor in config.get('sensors', []):
            # In production, use pymodbus to read actual registers
            value = 42.5  # Simulated value
            
            reading = SensorReading(
                machine_id=config['machine_id'],
                sensor_type=sensor['name'],
                value=value,
                timestamp=datetime.utcnow().isoformat()
            )
            readings.append(reading)
        
        return readings
    
    async def disconnect(self):
        self.connected = False
        logger.info("Modbus disconnected")

class MultiProtocolGateway:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.protocol_clients = {}
        self.running = False
    
    def add_protocol_client(self, name: str, client: ProtocolClient):
        self.protocol_clients[name] = client
    
    async def start(self, config_file: str):
        """Start gateway with device configuration"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update API endpoint from config if specified
        if 'api_endpoint' in config:
            self.api_endpoint = config['api_endpoint']
            logger.info(f"Using API endpoint from config: {self.api_endpoint}")
        
        # Update API key from config if specified
        if 'api_key' in config:
            self.api_key = config['api_key']
            logger.info("API key updated from config")
        
        self.running = True
        logger.info("Starting Multi-Protocol IoT Gateway v2.0")
        
        # Connect all clients
        for name, client in self.protocol_clients.items():
            await client.connect()
        
        # Start data collection loop
        await self._data_collection_loop(config)
    
    async def _data_collection_loop(self, config: Dict):
        """Main data collection loop"""
        while self.running:
            try:
                all_readings = []
                
                # Read from all devices
                for device in config.get('devices', []):
                    protocol = device.get('protocol')
                    if protocol in self.protocol_clients:
                        client = self.protocol_clients[protocol]
                        readings = await client.read_sensors(device)
                        all_readings.extend(readings)
                
                # Send to API
                if all_readings:
                    await self._send_to_api(all_readings)
                
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(5)
    
    async def _send_to_api(self, readings: List[SensorReading]):
        """Send readings to API"""
        try:
            async with aiohttp.ClientSession() as session:
                for reading in readings:
                    payload = {
                        "sensors": {
                            reading.sensor_type: reading.value
                        },
                        "timestamp": reading.timestamp
                    }
                    
                    headers = {
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json'
                    }
                    
                    async with session.post(
                        f"{self.api_endpoint}/api/v2/iot/data/{reading.machine_id}",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Sent data for {reading.machine_id}")
                        else:
                            logger.error(f"API error: {response.status}")
        
        except Exception as e:
            logger.error(f"API transmission error: {e}")
    
    async def stop(self):
        self.running = False
        for client in self.protocol_clients.values():
            await client.disconnect()

# Example usage
async def main():
    gateway = MultiProtocolGateway(
        api_endpoint="http://localhost:8001",
        api_key="your-api-key"
    )
    
    # Add protocol clients
    gateway.add_protocol_client("modbus", ModbusClient("192.168.1.100"))
    
    try:
        await gateway.start("config/device_configs/egypt_manufacturing.yaml")
    except KeyboardInterrupt:
        await gateway.stop()
        logger.info("Gateway stopped")

if __name__ == "__main__":
    asyncio.run(main())
