#!/usr/bin/env python3
"""
Equipment Configuration Generator for PDM Platform v2.0
Generates production-ready equipment configurations for real industrial protocols
"""

import json
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import argparse

@dataclass
class ModbusConfig:
    """Modbus equipment configuration"""
    host: str
    port: int = 502
    slave_id: int = 1
    timeout: int = 10
    connection_type: str = "tcp"  # tcp or serial
    # Serial specific (if connection_type = "serial")
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1

@dataclass
class OPCUAConfig:
    """OPC-UA equipment configuration"""
    endpoint: str
    security_mode: str = "None"
    username: str = None
    password: str = None
    certificate_path: str = None
    private_key_path: str = None

@dataclass
class MQTTConfig:
    """MQTT equipment configuration"""
    host: str
    port: int = 1883
    username: str = None
    password: str = None
    use_tls: bool = False
    keepalive: int = 60
    client_id: str = None

@dataclass
class SensorMapping:
    """Sensor mapping configuration"""
    # Modbus specific
    address: int = None
    register_type: str = None  # holding, input, coil, discrete
    data_type: str = None  # int16, uint16, int32, uint32, float32, float64
    count: int = 1
    byte_order: str = "big"
    # OPC-UA specific
    node_id: str = None
    # MQTT specific
    topic: str = None
    qos: int = 0
    max_age_seconds: int = 300
    # Common
    scale_factor: float = 1.0
    offset: float = 0.0
    unit: str = "unknown"

@dataclass
class EquipmentConfig:
    """Complete equipment configuration"""
    equipment_id: str
    protocol: str  # modbus, opcua, mqtt
    connection_params: Dict[str, Any]
    sensor_mappings: Dict[str, SensorMapping]
    tenant_id: str
    polling_interval: int = 30
    enabled: bool = True
    description: str = ""
    location: str = ""
    manufacturer: str = ""
    model: str = ""

class EquipmentConfigGenerator:
    """Generate equipment configurations for different industrial protocols"""
    
    def __init__(self):
        self.configurations = []
    
    def add_modbus_equipment(self, equipment_id: str, host: str, tenant_id: str, **kwargs) -> EquipmentConfig:
        """Add Modbus TCP equipment configuration"""
        
        # Default Modbus connection parameters
        modbus_config = ModbusConfig(host=host, **kwargs)
        
        # Standard sensor mappings for typical motor
        sensor_mappings = {
            "vibration_x": SensorMapping(
                address=0, register_type="holding", data_type="float32", 
                count=2, unit="mm/s", scale_factor=0.1
            ),
            "vibration_y": SensorMapping(
                address=2, register_type="holding", data_type="float32", 
                count=2, unit="mm/s", scale_factor=0.1
            ),
            "vibration_z": SensorMapping(
                address=4, register_type="holding", data_type="float32", 
                count=2, unit="mm/s", scale_factor=0.1
            ),
            "temperature": SensorMapping(
                address=6, register_type="holding", data_type="int16", 
                unit="°C", scale_factor=0.1
            ),
            "pressure": SensorMapping(
                address=7, register_type="holding", data_type="uint16", 
                unit="bar", scale_factor=0.01
            ),
            "speed_rpm": SensorMapping(
                address=8, register_type="holding", data_type="uint16", 
                unit="rpm", scale_factor=1.0
            ),
            "current_draw": SensorMapping(
                address=9, register_type="holding", data_type="uint16", 
                unit="A", scale_factor=0.1
            ),
            "power_consumption": SensorMapping(
                address=10, register_type="holding", data_type="uint16", 
                unit="kW", scale_factor=0.01
            )
        }
        
        config = EquipmentConfig(
            equipment_id=equipment_id,
            protocol="modbus",
            connection_params=asdict(modbus_config),
            sensor_mappings={k: asdict(v) for k, v in sensor_mappings.items()},
            tenant_id=tenant_id,
            description=f"Modbus TCP motor {equipment_id}",
            polling_interval=30
        )
        
        self.configurations.append(config)
        return config
    
    def add_opcua_equipment(self, equipment_id: str, endpoint: str, tenant_id: str, **kwargs) -> EquipmentConfig:
        """Add OPC-UA equipment configuration"""
        
        opcua_config = OPCUAConfig(endpoint=endpoint, **kwargs)
        
        # Standard OPC-UA node mappings
        base_node = f"ns=2;s={equipment_id}"
        sensor_mappings = {
            "vibration_x": SensorMapping(
                node_id=f"{base_node}.Vibration.X", 
                unit="mm/s", scale_factor=1.0
            ),
            "vibration_y": SensorMapping(
                node_id=f"{base_node}.Vibration.Y", 
                unit="mm/s", scale_factor=1.0
            ),
            "vibration_z": SensorMapping(
                node_id=f"{base_node}.Vibration.Z", 
                unit="mm/s", scale_factor=1.0
            ),
            "temperature": SensorMapping(
                node_id=f"{base_node}.Temperature", 
                unit="°C", scale_factor=1.0
            ),
            "pressure": SensorMapping(
                node_id=f"{base_node}.Pressure", 
                unit="bar", scale_factor=1.0
            ),
            "speed_rpm": SensorMapping(
                node_id=f"{base_node}.Speed", 
                unit="rpm", scale_factor=1.0
            ),
            "current_draw": SensorMapping(
                node_id=f"{base_node}.Current", 
                unit="A", scale_factor=1.0
            ),
            "power_consumption": SensorMapping(
                node_id=f"{base_node}.Power", 
                unit="kW", scale_factor=1.0
            )
        }
        
        config = EquipmentConfig(
            equipment_id=equipment_id,
            protocol="opcua",
            connection_params=asdict(opcua_config),
            sensor_mappings={k: asdict(v) for k, v in sensor_mappings.items()},
            tenant_id=tenant_id,
            description=f"OPC-UA connected equipment {equipment_id}",
            polling_interval=15
        )
        
        self.configurations.append(config)
        return config
    
    def add_mqtt_equipment(self, equipment_id: str, host: str, tenant_id: str, **kwargs) -> EquipmentConfig:
        """Add MQTT equipment configuration"""
        
        mqtt_config = MQTTConfig(host=host, **kwargs)
        
        # MQTT topic structure
        base_topic = f"factory/equipment/{equipment_id}"
        sensor_mappings = {
            "vibration_x": SensorMapping(
                topic=f"{base_topic}/vibration/x", 
                qos=1, unit="mm/s", scale_factor=1.0
            ),
            "vibration_y": SensorMapping(
                topic=f"{base_topic}/vibration/y", 
                qos=1, unit="mm/s", scale_factor=1.0
            ),
            "vibration_z": SensorMapping(
                topic=f"{base_topic}/vibration/z", 
                qos=1, unit="mm/s", scale_factor=1.0
            ),
            "temperature": SensorMapping(
                topic=f"{base_topic}/temperature", 
                qos=1, unit="°C", scale_factor=1.0
            ),
            "pressure": SensorMapping(
                topic=f"{base_topic}/pressure", 
                qos=1, unit="bar", scale_factor=1.0
            ),
            "speed_rpm": SensorMapping(
                topic=f"{base_topic}/speed", 
                qos=1, unit="rpm", scale_factor=1.0
            ),
            "current_draw": SensorMapping(
                topic=f"{base_topic}/current", 
                qos=1, unit="A", scale_factor=1.0
            ),
            "power_consumption": SensorMapping(
                topic=f"{base_topic}/power", 
                qos=1, unit="kW", scale_factor=1.0
            )
        }
        
        config = EquipmentConfig(
            equipment_id=equipment_id,
            protocol="mqtt",
            connection_params=asdict(mqtt_config),
            sensor_mappings={k: asdict(v) for k, v in sensor_mappings.items()},
            tenant_id=tenant_id,
            description=f"MQTT connected equipment {equipment_id}",
            polling_interval=10  # MQTT can be faster
        )
        
        self.configurations.append(config)
        return config
    
    def generate_sample_plant_config(self, tenant_name: str = "Main Plant") -> Dict[str, Any]:
        """Generate a complete sample plant configuration"""
        
        tenant_id = str(uuid.uuid4())
        
        # Clear existing configurations
        self.configurations.clear()
        
        # Add various equipment types
        
        # Modbus TCP Motors
        self.add_modbus_equipment(
            "EG_M001", "192.168.1.101", tenant_id,
            description="Main Motor #1", location="Building A - Floor 1",
            manufacturer="ABB", model="M3BP-160"
        )
        
        self.add_modbus_equipment(
            "EG_M002", "192.168.1.102", tenant_id,
            description="Main Motor #2", location="Building A - Floor 1", 
            manufacturer="Siemens", model="1LE1001"
        )
        
        self.add_modbus_equipment(
            "EG_M003", "192.168.1.103", tenant_id,
            description="Auxiliary Motor #1", location="Building A - Floor 2",
            manufacturer="WEG", model="W22"
        )
        
        # OPC-UA Equipment
        self.add_opcua_equipment(
            "PLC_001", "opc.tcp://192.168.1.201:4840", tenant_id,
            description="Main PLC Controller", location="Control Room",
            manufacturer="Schneider", model="M580"
        )
        
        self.add_opcua_equipment(
            "HMI_001", "opc.tcp://192.168.1.202:4840", tenant_id,
            description="Operator HMI", location="Control Room",
            manufacturer="Wonderware", model="InTouch"
        )
        
        # MQTT IoT Devices
        self.add_mqtt_equipment(
            "IOT_SENSOR_001", "192.168.1.250", tenant_id,
            description="Wireless Vibration Sensor", location="Building B",
            manufacturer="IFM", model="VVB001"
        )
        
        self.add_mqtt_equipment(
            "IOT_SENSOR_002", "192.168.1.250", tenant_id,
            description="Temperature Monitoring Node", location="Building B",
            manufacturer="Phoenix Contact", model="TC CLOUD"
        )
        
        # Additional Modbus RTU (Serial) equipment
        serial_config = self.add_modbus_equipment(
            "RTU_001", "dummy_host", tenant_id,
            connection_type="serial", serial_port="/dev/ttyUSB0",
            baudrate=9600, description="Serial RTU Device"
        )
        
        return {
            "tenant": {
                "id": tenant_id,
                "name": tenant_name,
                "description": f"Auto-generated configuration for {tenant_name}",
                "created_at": "2025-01-01T00:00:00Z"
            },
            "equipment": [asdict(config) for config in self.configurations]
        }
    
    def save_config(self, filename: str):
        """Save configuration to JSON file"""
        config_data = {
            "equipment": [asdict(config) for config in self.configurations]
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"✅ Configuration saved to {filename}")
        print(f"📊 Generated {len(self.configurations)} equipment configurations")
    
    def generate_docker_compose_override(self, config_file: str = "equipment_config.json") -> str:
        """Generate docker-compose override for protocol clients"""
        
        override_content = f"""
# Docker Compose override for PDM Platform Protocol Clients
version: '3.8'

services:
  protocol-clients:
    build:
      context: .
      dockerfile: docker/Dockerfile.protocols
    environment:
      - EQUIPMENT_CONFIG_FILE={config_file}
      - PDM_API_ENDPOINT=http://pdm-api:3000
      - LOG_LEVEL=info
    volumes:
      - ./{config_file}:/app/{config_file}:ro
      - ./logs:/app/logs
    depends_on:
      - pdm-api
    networks:
      - pdm-network
    restart: unless-stopped
    
  # Protocol client monitoring
  protocol-monitor:
    image: pdm-platform/monitoring:v2.0.1
    environment:
      - MONITOR_PROTOCOLS=true
      - CHECK_INTERVAL=60
    volumes:
      - ./logs:/app/logs:ro
    depends_on:
      - protocol-clients
    networks:
      - pdm-network
    restart: unless-stopped

networks:
  pdm-network:
    driver: bridge
"""
        
        return override_content.strip()
    
    def generate_kubernetes_configmap(self, config_file: str = "equipment_config.json") -> str:
        """Generate Kubernetes ConfigMap for equipment configuration"""
        
        configmap_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: equipment-config
  namespace: pdm-platform
  labels:
    app: pdm-platform
    component: protocol-clients
data:
  equipment_config.json: |
"""
        
        # Load and indent the configuration
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Indent each line by 4 spaces for YAML
            indented_content = '\n'.join('    ' + line for line in config_content.split('\n'))
            configmap_yaml += indented_content
            
        except FileNotFoundError:
            configmap_yaml += "    # Configuration file not found - generate with equipment_config_generator.py"
        
        return configmap_yaml
    
    def print_configuration_summary(self):
        """Print summary of generated configurations"""
        if not self.configurations:
            print("⚠️  No configurations generated")
            return
        
        print(f"\n📋 Equipment Configuration Summary")
        print("=" * 50)
        
        protocol_counts = {}
        for config in self.configurations:
            protocol = config.protocol
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        print(f"📊 Total Equipment: {len(self.configurations)}")
        for protocol, count in protocol_counts.items():
            print(f"   • {protocol.upper()}: {count} devices")
        
        print(f"\n📝 Equipment Details:")
        print("-" * 50)
        
        for config in self.configurations:
            print(f"🔧 {config.equipment_id} ({config.protocol.upper()})")
            print(f"   Description: {config.description}")
            print(f"   Sensors: {len(config.sensor_mappings)} configured")
            print(f"   Polling: {config.polling_interval}s")
            print(f"   Status: {'✅ Enabled' if config.enabled else '❌ Disabled'}")
            print()

def main():
    """Main configuration generator CLI"""
    parser = argparse.ArgumentParser(description='PDM Platform Equipment Configuration Generator')
    parser.add_argument('--output', '-o', default='equipment_config.json', 
                        help='Output configuration file')
    parser.add_argument('--tenant-name', default='Main Plant', 
                        help='Tenant name for generated configuration')
    parser.add_argument('--generate-sample', action='store_true', 
                        help='Generate sample plant configuration')
    parser.add_argument('--docker-compose', action='store_true',
                        help='Generate Docker Compose override')
    parser.add_argument('--kubernetes', action='store_true',
                        help='Generate Kubernetes ConfigMap')
    parser.add_argument('--custom', action='store_true',
                        help='Interactive custom configuration')
    
    args = parser.parse_args()
    
    generator = EquipmentConfigGenerator()
    
    if args.generate_sample:
        print("🏭 Generating sample plant configuration...")
        config = generator.generate_sample_plant_config(args.tenant_name)
        
        # Save main configuration
        generator.save_config(args.output)
        
        # Save tenant information separately
        with open('tenant_config.json', 'w') as f:
            json.dump(config['tenant'], f, indent=2)
        
        print("✅ Tenant configuration saved to tenant_config.json")
        
    elif args.custom:
        print("🛠️  Interactive Configuration Mode")
        print("=" * 40)
        
        while True:
            print("\nAdd Equipment:")
            print("1. Modbus TCP Motor")
            print("2. OPC-UA Device") 
            print("3. MQTT IoT Sensor")
            print("4. Finish and Save")
            
            choice = input("Choose option (1-4): ").strip()
            
            if choice == "1":
                equipment_id = input("Equipment ID: ").strip()
                host = input("Modbus Host IP: ").strip()
                tenant_id = input("Tenant ID (or press Enter for auto): ").strip()
                if not tenant_id:
                    tenant_id = str(uuid.uuid4())
                
                generator.add_modbus_equipment(equipment_id, host, tenant_id)
                print(f"✅ Added Modbus equipment: {equipment_id}")
                
            elif choice == "2":
                equipment_id = input("Equipment ID: ").strip()
                endpoint = input("OPC-UA Endpoint: ").strip()
                tenant_id = input("Tenant ID (or press Enter for auto): ").strip()
                if not tenant_id:
                    tenant_id = str(uuid.uuid4())
                
                generator.add_opcua_equipment(equipment_id, endpoint, tenant_id)
                print(f"✅ Added OPC-UA equipment: {equipment_id}")
                
            elif choice == "3":
                equipment_id = input("Equipment ID: ").strip()
                host = input("MQTT Broker Host: ").strip()
                tenant_id = input("Tenant ID (or press Enter for auto): ").strip()
                if not tenant_id:
                    tenant_id = str(uuid.uuid4())
                
                generator.add_mqtt_equipment(equipment_id, host, tenant_id)
                print(f"✅ Added MQTT equipment: {equipment_id}")
                
            elif choice == "4":
                if generator.configurations:
                    generator.save_config(args.output)
                else:
                    print("⚠️  No equipment configured")
                break
            else:
                print("❌ Invalid choice")
    
    # Generate additional files
    if args.docker_compose and generator.configurations:
        override_content = generator.generate_docker_compose_override(args.output)
        with open('docker-compose.protocols.yml', 'w') as f:
            f.write(override_content)
        print("✅ Docker Compose override saved to docker-compose.protocols.yml")
    
    if args.kubernetes and generator.configurations:
        configmap_content = generator.generate_kubernetes_configmap(args.output)
        with open('k8s-equipment-configmap.yaml', 'w') as f:
            f.write(configmap_content)
        print("✅ Kubernetes ConfigMap saved to k8s-equipment-configmap.yaml")
    
    # Print summary
    generator.print_configuration_summary()
    
    if generator.configurations:
        print(f"\n🚀 Next Steps:")
        print(f"1. Review and customize {args.output}")
        print(f"2. Test connectivity to configured devices")
        print(f"3. Deploy protocol clients with configuration")
        print(f"4. Monitor data ingestion in PDM Platform dashboard")

if __name__ == "__main__":
    main()
