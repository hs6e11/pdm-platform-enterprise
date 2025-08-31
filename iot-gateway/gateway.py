# iot-gateway/gateway.py
"""
Production IoT Gateway for PDM Platform v2.0
Integrates real industrial protocols with multi-tenant API
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
from pathlib import Path
import yaml
import signal
import sys
import os
from dataclasses import dataclass
import time

# Import our real protocol clients
from protocols.real_clients import create_protocol_client, ProtocolError, BaseProtocolClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gateway.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GatewayConfig:
    """Gateway configuration data class"""
    api_url: str
    api_key: str
    tenant_id: str
    poll_interval: int = 30
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: int = 5
    health_check_interval: int = 60

@dataclass
class ProtocolConfig:
    """Protocol-specific configuration"""
    protocol_type: str
    enabled: bool
    config: Dict[str, Any]

class GatewayStats:
    """Track gateway performance statistics"""
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.readings_collected = 0
        self.readings_sent = 0
        self.api_errors = 0
        self.protocol_errors = 0
        self.last_successful_send = None
        self.uptime_checks = 0
        self.health_status = "starting"
    
    def log_stats(self):
        uptime = datetime.utcnow() - self.start_time
        success_rate = (self.readings_sent / max(self.readings_collected, 1)) * 100
        
        logger.info(f"""
Gateway Statistics:
==================
Uptime: {uptime.total_seconds():.0f} seconds
Readings collected: {self.readings_collected}
Readings sent: {self.readings_sent}
Success rate: {success_rate:.1f}%
API errors: {self.api_errors}
Protocol errors: {self.protocol_errors}
Health status: {self.health_status}
Last successful send: {self.last_successful_send}
        """)

class IoTGateway:
    """Production IoT Gateway with real protocol integration"""
    
    def __init__(self, config_path: str = "config/gateway_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.protocol_clients = {}
        self.stats = GatewayStats()
        self.running = False
        self.session = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"IoT Gateway initialized with config: {config_path}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def load_configuration(self) -> bool:
        """Load gateway configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse gateway configuration
            gateway_config = config_data.get('gateway', {})
            self.config = GatewayConfig(
                api_url=gateway_config['api_url'],
                api_key=gateway_config['api_key'],
                tenant_id=gateway_config['tenant_id'],
                poll_interval=gateway_config.get('poll_interval', 30),
                batch_size=gateway_config.get('batch_size', 100),
                max_retries=gateway_config.get('max_retries', 3),
                retry_delay=gateway_config.get('retry_delay', 5),
                health_check_interval=gateway_config.get('health_check_interval', 60)
            )
            
            # Parse protocol configurations
            protocols_config = config_data.get('protocols', {})
            self.protocol_configs = {}
            
            for protocol_name, protocol_data in protocols_config.items():
                if protocol_data.get('enabled', False):
                    self.protocol_configs[protocol_name] = ProtocolConfig(
                        protocol_type=protocol_data['type'],
                        enabled=True,
                        config=protocol_data.get('config', {})
                    )
            
            logger.info(f"Configuration loaded successfully. Enabled protocols: {list(self.protocol_configs.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    async def initialize_protocols(self) -> bool:
        """Initialize all configured protocol clients"""
        logger.info("Initializing protocol clients...")
        
        for protocol_name, protocol_config in self.protocol_configs.items():
            try:
                # Create protocol client
                client = create_protocol_client(protocol_config.protocol_type, protocol_config.config)
                
                # Attempt to connect
                if await client.connect():
                    self.protocol_clients[protocol_name] = client
                    logger.info(f"✅ {protocol_name} protocol initialized successfully")
                else:
                    logger.error(f"❌ Failed to initialize {protocol_name} protocol")
                    
            except Exception as e:
                logger.error(f"Error initializing {protocol_name} protocol: {e}")
                self.stats.protocol_errors += 1
        
        if not self.protocol_clients:
            logger.error("No protocol clients initialized successfully")
            return False
        
        logger.info(f"Initialized {len(self.protocol_clients)} protocol clients")
        return True
    
    async def collect_sensor_data(self) -> List[Dict[str, Any]]:
        """Collect sensor data from all active protocol clients"""
        all_readings = []
        collection_timestamp = datetime.utcnow()
        
        for protocol_name, client in self.protocol_clients.items():
            try:
                if not client.is_connected:
                    logger.warning(f"Protocol {protocol_name} is disconnected, attempting reconnection...")
                    if not await client.connect():
                        logger.error(f"Failed to reconnect {protocol_name}")
                        self.stats.protocol_errors += 1
                        continue
                
                # Collect readings from this protocol
                readings = await client.read_sensors()
                
                # Add gateway metadata to each reading
                for reading in readings:
                    reading['gateway_id'] = self.config.tenant_id
                    reading['collection_timestamp'] = collection_timestamp
                    reading['protocol_client'] = protocol_name
                
                all_readings.extend(readings)
                logger.debug(f"Collected {len(readings)} readings from {protocol_name}")
                
            except ProtocolError as e:
                logger.error(f"Protocol error in {protocol_name}: {e}")
                self.stats.protocol_errors += 1
                
                # Mark client as disconnected and try to reconnect next cycle
                client.is_connected = False
                
            except Exception as e:
                logger.error(f"Unexpected error collecting from {protocol_name}: {e}")
                self.stats.protocol_errors += 1
        
        self.stats.readings_collected += len(all_readings)
        logger.info(f"Collected {len(all_readings)} sensor readings from {len(self.protocol_clients)} protocols")
        
        return all_readings
    
    async def send_data_to_api(self, readings: List[Dict[str, Any]]) -> bool:
        """Send sensor data to Phase 2 API with retry logic"""
        if not readings:
            return True
        
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'X-Gateway-Client': 'PDM-IoT-Gateway-v2.0'
        }
        
        # Prepare data for API
        api_data = {
            'tenant_id': self.config.tenant_id,
            'readings': readings,
            'batch_metadata': {
                'collection_timestamp': datetime.utcnow().isoformat(),
                'gateway_version': '2.0',
                'batch_size': len(readings)
            }
        }
        
        # Retry logic
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self.session.post(
                    f"{self.config.api_url}/api/v2/sensor-data/batch",
                    json=api_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        self.stats.readings_sent += len(readings)
                        self.stats.last_successful_send = datetime.utcnow()
                        
                        logger.info(f"✅ Successfully sent {len(readings)} readings to API")
                        return True
                        
                    elif response.status == 401:
                        logger.error("❌ Authentication failed - check API key and tenant ID")
                        self.stats.api_errors += 1
                        return False
                        
                    elif response.status == 429:
                        logger.warning("⚠️ Rate limited by API, will retry...")
                        await asyncio.sleep(self.config.retry_delay * attempt)
                        continue
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ API returned status {response.status}: {error_text}")
                        
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * attempt)
                            continue
                        else:
                            self.stats.api_errors += 1
                            return False
            
            except asyncio.TimeoutError:
                logger.error(f"⏱️ API request timeout (attempt {attempt}/{self.config.max_retries})")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                    continue
                else:
                    self.stats.api_errors += 1
                    return False
            
            except Exception as e:
                logger.error(f"❌ API request error (attempt {attempt}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                    continue
                else:
                    self.stats.api_errors += 1
                    return False
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': (datetime.utcnow() - self.stats.start_time).total_seconds(),
            'protocols': {},
            'stats': {
                'readings_collected': self.stats.readings_collected,
                'readings_sent': self.stats.readings_sent,
                'api_errors': self.stats.api_errors,
                'protocol_errors': self.stats.protocol_errors,
                'success_rate': (self.stats.readings_sent / max(self.stats.readings_collected, 1)) * 100
            },
            'status': 'healthy'
        }
        
        # Check protocol client health
        unhealthy_protocols = 0
        for protocol_name, client in self.protocol_clients.items():
            protocol_health = {
                'connected': client.is_connected,
                'last_error': client.last_error,
                'connection_attempts': client.connection_attempts
            }
            health_data['protocols'][protocol_name] = protocol_health
            
            if not client.is_connected:
                unhealthy_protocols += 1
        
        # Determine overall health status
        if unhealthy_protocols == len(self.protocol_clients):
            health_data['status'] = 'critical'
        elif unhealthy_protocols > 0:
            health_data['status'] = 'degraded'
        elif self.stats.api_errors > 0 and self.stats.last_successful_send:
            time_since_success = datetime.utcnow() - self.stats.last_successful_send
            if time_since_success > timedelta(minutes=10):
                health_data['status'] = 'degraded'
        
        self.stats.health_status = health_data['status']
        self.stats.uptime_checks += 1
        
        return health_data
    
    async def send_health_check(self):
        """Send health check data to API"""
        try:
            health_data = await self.health_check()
            
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                f"{self.config.api_url}/api/v2/gateway/health",
                json=health_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    logger.debug("Health check sent successfully")
                else:
                    logger.warning(f"Health check failed: {response.status}")
                    
        except Exception as e:
            logger.warning(f"Failed to send health check: {e}")
    
    async def main_loop(self):
        """Main gateway operation loop"""
        logger.info("Starting main gateway loop...")
        
        last_health_check = datetime.utcnow()
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Collect sensor data
                readings = await self.collect_sensor_data()
                
                # Send data to API in batches
                if readings:
                    # Process in batches if we have a lot of readings
                    for i in range(0, len(readings), self.config.batch_size):
                        batch = readings[i:i + self.config.batch_size]
                        await self.send_data_to_api(batch)
                
                # Send health check periodically
                now = datetime.utcnow()
                if (now - last_health_check).total_seconds() >= self.config.health_check_interval:
                    await self.send_health_check()
                    last_health_check = now
                
                # Log statistics periodically
                if self.stats.uptime_checks % 10 == 0:
                    self.stats.log_stats()
                
                # Calculate sleep time to maintain polling interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.poll_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Loop took {loop_duration:.1f}s, longer than poll interval {self.config.poll_interval}s")
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def start(self):
        """Start the IoT Gateway"""
        logger.info("=== PDM Platform v2.0 IoT Gateway Starting ===")
        
        # Load configuration
        if not self.load_configuration():
            logger.error("Failed to load configuration, exiting")
            return False
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize protocol clients
        if not await self.initialize_protocols():
            logger.error("Failed to initialize protocol clients, exiting")
            await self.session.close()
            return False
        
        # Start main operation
        self.running = True
        self.stats.health_status = "running"
        
        try:
            logger.info("🚀 Gateway started successfully!")
            await self.main_loop()
            
        except Exception as e:
            logger.error(f"Gateway error: {e}")
            return False
        
        finally:
            await self.shutdown()
        
        return True
    
    async def shutdown(self):
        """Shutdown gateway gracefully"""
        logger.info("Shutting down gateway...")
        
        self.running = False
        self.stats.health_status = "stopping"
        
        # Disconnect protocol clients
        for protocol_name, client in self.protocol_clients.items():
            try:
                await client.disconnect()
                logger.info(f"Disconnected {protocol_name} protocol")
            except Exception as e:
                logger.warning(f"Error disconnecting {protocol_name}: {e}")
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Log final statistics
        self.stats.log_stats()
        
        logger.info("Gateway shutdown complete")

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        'gateway': {
            'api_url': 'http://localhost:8001',
            'api_key': 'your_tenant_api_key_here',
            'tenant_id': 'your_tenant_uuid_here',
            'poll_interval': 30,
            'batch_size': 100,
            'max_retries': 3,
            'retry_delay': 5,
            'health_check_interval': 60
        },
        'protocols': {
            'modbus_plc1': {
                'type': 'modbus',
                'enabled': True,
                'config': {
                    'client_id': 'modbus_client_1',
                    'host': '192.168.1.100',
                    'port': 502,
                    'unit_id': 1,
                    'type': 'tcp',
                    'registers': {
                        'EG_M001': {
                            'sensors': [
                                {
                                    'address': 1000,
                                    'type': 'temperature',
                                    'register_type': 'holding',
                                    'scale_factor': 0.1
                                },
                                {
                                    'address': 1001,
                                    'type': 'spindle_speed',
                                    'register_type': 'holding',
                                    'scale_factor': 1.0
                                }
                            ]
                        }
                    }
                }
            },
            'opcua_server1': {
                'type': 'opcua',
                'enabled': False,
                'config': {
                    'client_id': 'opcua_client_1',
                    'endpoint_url': 'opc.tcp://192.168.1.101:4840',
                    'security_policy': 'Basic256Sha256',
                    'security_mode': 'SignAndEncrypt',
                    'nodes': {
                        'EG_M002': {
                            'sensors': [
                                {
                                    'node_id': 'ns=2;i=1000',
                                    'type': 'conveyor_speed'
                                }
                            ]
                        }
                    }
                }
            },
            'mqtt_sensors': {
                'type': 'mqtt',
                'enabled': False,
                'config': {
                    'client_id': 'mqtt_client_1',
                    'host': '192.168.1.102',
                    'port': 1883,
                    'username': 'mqtt_user',
                    'password': 'mqtt_password',
                    'topics': {
                        'EG_M003': {
                            'sensors': [
                                {
                                    'topic': 'factory/machine3/pressure',
                                    'type': 'pressure',
                                    'qos': 1,
                                    'value_path': 'value'
                                }
                            ]
                        }
                    }
                }
            }
        }
    }
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'gateway_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    logger.info("Sample configuration created at config/gateway_config.yaml")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDM Platform v2.0 IoT Gateway')
    parser.add_argument('--config', default='config/gateway_config.yaml', help='Configuration file path')
    parser.add_argument('--create-sample-config', action='store_true', help='Create sample configuration file')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Start the gateway
    gateway = IoTGateway(args.config)
    success = await gateway.start()
    
    if not success:
        logger.error("Gateway failed to start")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
