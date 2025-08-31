#!/usr/bin/env python3
"""
Production Monitoring & Alerting System
Real-time system health monitoring with multi-channel notifications
"""

import asyncio
import asyncpg
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import ssl
import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class Alert:
    id: str
    tenant_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    equipment_id: Optional[str] = None
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SystemMetric:
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class NotificationChannel:
    name: str
    type: str  # email, slack, webhook, sms
    config: Dict[str, Any]
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.LOW

class SystemMonitor:
    """System health monitoring with metrics collection"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[SystemMetric]] = {}
        self.last_collection = None
        
    async def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now(timezone.utc)
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            metrics.extend([
                SystemMetric("cpu_usage_percent", cpu_percent, "%", timestamp, {"type": "system"}),
                SystemMetric("cpu_count", cpu_count, "cores", timestamp, {"type": "system"}),
                SystemMetric("load_average", load_avg, "load", timestamp, {"type": "system"})
            ])
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                SystemMetric("memory_total", memory.total / (1024**3), "GB", timestamp, {"type": "memory"}),
                SystemMetric("memory_used", memory.used / (1024**3), "GB", timestamp, {"type": "memory"}),
                SystemMetric("memory_percent", memory.percent, "%", timestamp, {"type": "memory"}),
                SystemMetric("swap_total", swap.total / (1024**3), "GB", timestamp, {"type": "memory"}),
                SystemMetric("swap_used", swap.used / (1024**3), "GB", timestamp, {"type": "memory"}),
                SystemMetric("swap_percent", swap.percent, "%", timestamp, {"type": "memory"})
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.extend([
                SystemMetric("disk_total", disk_usage.total / (1024**3), "GB", timestamp, {"type": "disk"}),
                SystemMetric("disk_used", disk_usage.used / (1024**3), "GB", timestamp, {"type": "disk"}),
                SystemMetric("disk_percent", (disk_usage.used / disk_usage.total) * 100, "%", timestamp, {"type": "disk"}),
                SystemMetric("disk_read_bytes", disk_io.read_bytes / (1024**2), "MB", timestamp, {"type": "disk"}),
                SystemMetric("disk_write_bytes", disk_io.write_bytes / (1024**2), "MB", timestamp, {"type": "disk"})
            ])
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            metrics.extend([
                SystemMetric("network_sent", net_io.bytes_sent / (1024**2), "MB", timestamp, {"type": "network"}),
                SystemMetric("network_recv", net_io.bytes_recv / (1024**2), "MB", timestamp, {"type": "network"}),
                SystemMetric("network_packets_sent", net_io.packets_sent, "packets", timestamp, {"type": "network"}),
                SystemMetric("network_packets_recv", net_io.packets_recv, "packets", timestamp, {"type": "network"})
            ])
            
            # Process metrics
            process_count = len(psutil.pids())
            
            metrics.append(
                SystemMetric("process_count", process_count, "processes", timestamp, {"type": "system"})
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
        
        # Store metrics in history
        for metric in metrics:
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append(metric)
            
            # Keep only last 1000 measurements
            if len(self.metrics_history[metric.name]) > 1000:
                self.metrics_history[metric.name] = self.metrics_history[metric.name][-1000:]
        
        self.last_collection = timestamp
        logger.info(f"Collected {len(metrics)} system metrics")
        return metrics
    
    async def collect_database_metrics(self, db_url: str) -> List[SystemMetric]:
        """Collect database-specific metrics"""
        metrics = []
        timestamp = datetime.now(timezone.utc)
        
        try:
            conn = await asyncpg.connect(db_url)
            
            # Connection count
            conn_count = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            )
            
            # Database size
            db_size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            
            # Convert size to MB (rough conversion)
            try:
                if 'GB' in db_size:
                    size_mb = float(db_size.split()[0]) * 1024
                elif 'MB' in db_size:
                    size_mb = float(db_size.split()[0])
                else:
                    size_mb = 0.0
            except:
                size_mb = 0.0
            
            # Recent readings count
            recent_readings = await conn.fetchval(
                "SELECT COUNT(*) FROM sensor_readings WHERE timestamp >= NOW() - INTERVAL '1 hour'"
            )
            
            # Active anomalies count
            active_anomalies = await conn.fetchval(
                "SELECT COUNT(*) FROM anomalies WHERE created_at >= NOW() - INTERVAL '1 hour'"
            )
            
            # Table sizes
            table_sizes = await conn.fetch("""
                SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 10
            """)
            
            metrics.extend([
                SystemMetric("db_connections_active", conn_count, "connections", timestamp, {"type": "database"}),
                SystemMetric("db_size", size_mb, "MB", timestamp, {"type": "database"}),
                SystemMetric("db_recent_readings", recent_readings, "records", timestamp, {"type": "database"}),
                SystemMetric("db_active_anomalies", active_anomalies, "anomalies", timestamp, {"type": "database"})
            ])
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
        
        return metrics
    
    async def collect_application_metrics(self, api_endpoint: str) -> List[SystemMetric]:
        """Collect application-specific metrics"""
        metrics = []
        timestamp = datetime.now(timezone.utc)
        
        try:
            async with aiohttp.ClientSession() as session:
                # API health check
                start_time = time.time()
                async with session.get(f"{api_endpoint}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    api_status = 1 if response.status == 200 else 0
                    
                    metrics.extend([
                        SystemMetric("api_response_time", response_time, "ms", timestamp, {"type": "application"}),
                        SystemMetric("api_status", api_status, "status", timestamp, {"type": "application"})
                    ])
                
                # Try to get application metrics if endpoint exists
                try:
                    async with session.get(f"{api_endpoint}/metrics") as response:
                        if response.status == 200:
                            app_metrics = await response.json()
                            
                            for metric_name, value in app_metrics.items():
                                if isinstance(value, (int, float)):
                                    metrics.append(
                                        SystemMetric(f"app_{metric_name}", value, "count", timestamp, {"type": "application"})
                                    )
                except:
                    pass  # Metrics endpoint might not exist
                    
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
        
        return metrics

class AlertManager:
    """Alert management with deduplication and escalation"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[NotificationChannel] = []
        self.cooldown_period = 300  # 5 minutes between same alerts
        
    def add_alert_rule(self, name: str, condition: Dict[str, Any]):
        """Add alert rule for monitoring"""
        self.alert_rules[name] = condition
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel.name} ({channel.type})")
    
    async def evaluate_alert_rules(self, metrics: List[SystemMetric]) -> List[Alert]:
        """Evaluate metrics against alert rules"""
        new_alerts = []
        
        # Group metrics by name for easier lookup
        metric_map = {m.name: m for m in metrics}
        
        for rule_name, rule_config in self.alert_rules.items():
            try:
                metric_name = rule_config['metric']
                operator = rule_config['operator']  # gt, lt, gte, lte, eq
                threshold = rule_config['threshold']
                severity = AlertSeverity(rule_config.get('severity', 'medium'))
                
                if metric_name in metric_map:
                    metric = metric_map[metric_name]
                    
                    # Evaluate condition
                    triggered = False
                    if operator == 'gt' and metric.value > threshold:
                        triggered = True
                    elif operator == 'lt' and metric.value < threshold:
                        triggered = True
                    elif operator == 'gte' and metric.value >= threshold:
                        triggered = True
                    elif operator == 'lte' and metric.value <= threshold:
                        triggered = True
                    elif operator == 'eq' and metric.value == threshold:
                        triggered = True
                    
                    if triggered:
                        # Create alert
                        alert_id = self._generate_alert_id(rule_name, metric_name)
                        
                        # Check if this alert already exists and is in cooldown
                        if self._is_in_cooldown(alert_id):
                            continue
                        
                        alert = Alert(
                            id=alert_id,
                            tenant_id="system",  # System alerts
                            alert_type="metric_threshold",
                            severity=severity,
                            title=f"{rule_config.get('title', rule_name)} Alert",
                            description=f"{metric_name} is {metric.value}{metric.unit} (threshold: {threshold}{metric.unit})",
                            metric_name=metric_name,
                            current_value=metric.value,
                            threshold_value=threshold,
                            metadata={
                                'rule_name': rule_name,
                                'operator': operator,
                                'metric_tags': metric.tags
                            }
                        )
                        
                        new_alerts.append(alert)
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {str(e)}")
        
        return new_alerts
    
    def _generate_alert_id(self, rule_name: str, metric_name: str) -> str:
        """Generate consistent alert ID for deduplication"""
        content = f"{rule_name}_{metric_name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _is_in_cooldown(self, alert_id: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_id in self.active_alerts:
            last_alert = self.active_alerts[alert_id]
            time_since = (datetime.now(timezone.utc) - last_alert.updated_at).total_seconds()
            return time_since < self.cooldown_period
        return False
    
    async def process_alerts(self, alerts: List[Alert]):
        """Process new alerts - store in DB and send notifications"""
        for alert in alerts:
            try:
                # Store in database
                await self._store_alert(alert)
                
                # Add to active alerts
                self.active_alerts[alert.id] = alert
                
                # Send notifications
                await self._send_notifications(alert)
                
                logger.info(f"Processed alert: {alert.title} ({alert.severity.value})")
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.id}: {str(e)}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = await asyncpg.connect(self.db_url)
            
            await conn.execute("""
                INSERT INTO system_alerts (
                    id, tenant_id, alert_type, severity, title, description,
                    equipment_id, metric_name, current_value, threshold_value,
                    created_at, status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = $11,
                    current_value = $9,
                    metadata = $13
            """, 
            alert.id, alert.tenant_id, alert.alert_type, alert.severity.value,
            alert.title, alert.description, alert.equipment_id, alert.metric_name,
            alert.current_value, alert.threshold_value, alert.created_at,
            alert.status.value, json.dumps(alert.metadata)
            )
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error storing alert in database: {str(e)}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        for channel in self.notification_channels:
            if not channel.enabled:
                continue
            
            # Check severity filter
            severity_levels = {
                AlertSeverity.LOW: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.HIGH: 3,
                AlertSeverity.CRITICAL: 4
            }
            
            if severity_levels[alert.severity] < severity_levels[channel.min_severity]:
                continue
            
            try:
                if channel.type == 'email':
                    await self._send_email_notification(channel, alert)
                elif channel.type == 'slack':
                    await self._send_slack_notification(channel, alert)
                elif channel.type == 'webhook':
                    await self._send_webhook_notification(channel, alert)
                # Add SMS, Teams, etc. as needed
                
            except Exception as e:
                logger.error(f"Error sending notification via {channel.name}: {str(e)}")
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert):
        """Send email notification"""
        config = channel.config
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] PDM Platform Alert: {alert.title}"
        
        # HTML body
        body = f"""
        <html>
        <body>
            <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.HIGH else 'blue'};">
                {alert.title}
            </h2>
            
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            
            {f'<p><strong>Equipment:</strong> {alert.equipment_id}</p>' if alert.equipment_id else ''}
            {f'<p><strong>Metric:</strong> {alert.metric_name}</p>' if alert.metric_name else ''}
            {f'<p><strong>Current Value:</strong> {alert.current_value}</p>' if alert.current_value is not None else ''}
            {f'<p><strong>Threshold:</strong> {alert.threshold_value}</p>' if alert.threshold_value is not None else ''}
            
            <p><strong>Description:</strong></p>
            <p>{alert.description}</p>
            
            <hr>
            <p><em>This is an automated message from PDM Platform Monitoring System</em></p>
        </body>
        </html>
        """
        
        msg.attach(MimeText(body, 'html'))
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
            server.starttls(context=context)
            server.login(config['smtp_user'], config['smtp_password'])
            server.send_message(msg)
        
        logger.info(f"Email alert sent to {len(config['to_emails'])} recipients")
    
    async def _send_slack_notification(self, channel: NotificationChannel, alert: Alert):
        """Send Slack notification"""
        config = channel.config
        webhook_url = config['webhook_url']
        
        # Color based on severity
        colors = {
            AlertSeverity.LOW: "#36a64f",      # green
            AlertSeverity.MEDIUM: "#ff9500",   # orange
            AlertSeverity.HIGH: "#ff0000",     # red
            AlertSeverity.CRITICAL: "#8B0000"  # dark red
        }
        
        payload = {
            "text": f"PDM Platform Alert: {alert.title}",
            "attachments": [
                {
                    "color": colors.get(alert.severity, "#36a64f"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                    ] + 
                    ([{"title": "Equipment", "value": alert.equipment_id, "short": True}] if alert.equipment_id else []) +
                    ([{"title": "Metric", "value": alert.metric_name, "short": True}] if alert.metric_name else []) +
                    ([{"title": "Current Value", "value": str(alert.current_value), "short": True}] if alert.current_value is not None else []) +
                    ([{"title": "Threshold", "value": str(alert.threshold_value), "short": True}] if alert.threshold_value is not None else []) +
                    [{"title": "Description", "value": alert.description, "short": False}]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack alert sent successfully")
                else:
                    logger.error(f"Failed to send Slack alert: {response.status}")
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert):
        """Send webhook notification"""
        config = channel.config
        url = config['url']
        
        payload = {
            "alert": asdict(alert),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "pdm_platform_monitoring"
        }
        
        headers = {'Content-Type': 'application/json'}
        if 'headers' in config:
            headers.update(config['headers'])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Webhook alert sent to {url}")
                else:
                    logger.error(f"Failed to send webhook alert: {response.status}")

class ProductionMonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self, db_url: str, api_endpoint: str):
        self.db_url = db_url
        self.api_endpoint = api_endpoint
        self.system_monitor = SystemMonitor()
        self.alert_manager = AlertManager(db_url)
        self.running = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        # Initialize notification channels from environment
        self._setup_notification_channels()
    
    def _setup_default_alert_rules(self):
        """Setup default system alert rules"""
        default_rules = [
            {
                'name': 'high_cpu_usage',
                'metric': 'cpu_usage_percent',
                'operator': 'gt',
                'threshold': 80,
                'severity': 'high',
                'title': 'High CPU Usage'
            },
            {
                'name': 'high_memory_usage',
                'metric': 'memory_percent',
                'operator': 'gt',
                'threshold': 85,
                'severity': 'high',
                'title': 'High Memory Usage'
            },
            {
                'name': 'high_disk_usage',
                'metric': 'disk_percent',
                'operator': 'gt',
                'threshold': 90,
                'severity': 'critical',
                'title': 'High Disk Usage'
            },
            {
                'name': 'api_down',
                'metric': 'api_status',
                'operator': 'eq',
                'threshold': 0,
                'severity': 'critical',
                'title': 'API Service Down'
            },
            {
                'name': 'slow_api_response',
                'metric': 'api_response_time',
                'operator': 'gt',
                'threshold': 5000,  # 5 seconds
                'severity': 'medium',
                'title': 'Slow API Response'
            }
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule['name'], rule)
    
    def _setup_notification_channels(self):
        """Setup notification channels from environment variables"""
        # Email notifications
        if all(os.getenv(var) for var in ['SMTP_HOST', 'SMTP_USER', 'SMTP_PASS', 'ALERT_EMAIL_TO']):
            email_channel = NotificationChannel(
                name="email_alerts",
                type="email",
                config={
                    'smtp_host': os.getenv('SMTP_HOST'),
                    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                    'smtp_user': os.getenv('SMTP_USER'),
                    'smtp_password': os.getenv('SMTP_PASS'),
                    'from_email': os.getenv('SMTP_USER'),
                    'to_emails': os.getenv('ALERT_EMAIL_TO').split(',')
                },
                min_severity=AlertSeverity.MEDIUM
            )
            self.alert_manager.add_notification_channel(email_channel)
        
        # Slack notifications
        if os.getenv('SLACK_WEBHOOK_URL'):
            slack_channel = NotificationChannel(
                name="slack_alerts",
                type="slack",
                config={
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL')
                },
                min_severity=AlertSeverity.HIGH
            )
            self.alert_manager.add_notification_channel(slack_channel)
        
        # Webhook notifications
        if os.getenv('WEBHOOK_ALERT_URL'):
            webhook_channel = NotificationChannel(
                name="webhook_alerts",
                type="webhook",
                config={
                    'url': os.getenv('WEBHOOK_ALERT_URL'),
                    'headers': json.loads(os.getenv('WEBHOOK_HEADERS', '{}'))
                },
                min_severity=AlertSeverity.LOW
            )
            self.alert_manager.add_notification_channel(webhook_channel)
    
    async def initialize_database_tables(self):
        """Create necessary database tables"""
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # System metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(100) NOT NULL,
                    value DECIMAL(15,4) NOT NULL,
                    unit VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tags JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # System alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id VARCHAR(50) PRIMARY KEY,
                    tenant_id UUID NOT NULL,
                    alert_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    description TEXT,
                    equipment_id VARCHAR(50),
                    metric_name VARCHAR(100),
                    current_value DECIMAL(15,4),
                    threshold_value DECIMAL(15,4),
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(20) DEFAULT 'active',
                    metadata JSONB
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(name, timestamp DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_system_alerts_severity_time ON system_alerts(severity, created_at DESC)")
            
            await conn.close()
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
    
    async def store_metrics(self, metrics: List[SystemMetric]):
        """Store metrics in database"""
        if not metrics:
            return
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Batch insert metrics
            values = [
                (m.name, m.value, m.unit, m.timestamp, json.dumps(m.tags))
                for m in metrics
            ]
            
            await conn.executemany(
                "INSERT INTO system_metrics (name, value, unit, timestamp, tags) VALUES ($1, $2, $3, $4, $5)",
                values
            )
            
            await conn.close()
            logger.info(f"Stored {len(metrics)} metrics in database")
            
        except Exception as e:
            logger.error(f"Error storing metrics: {str(e)}")
    
    async def run_monitoring_cycle(self):
        """Run single monitoring cycle"""
        try:
            # Collect all metrics
            system_metrics = await self.system_monitor.collect_system_metrics()
            db_metrics = await self.system_monitor.collect_database_metrics(self.db_url)
            app_metrics = await self.system_monitor.collect_application_metrics(self.api_endpoint)
            
            all_metrics = system_metrics + db_metrics + app_metrics
            
            # Store metrics in database
            await self.store_metrics(all_metrics)
            
            # Evaluate alert rules
            new_alerts = await self.alert_manager.evaluate_alert_rules(all_metrics)
            
            # Process alerts
            if new_alerts:
                await self.alert_manager.process_alerts(new_alerts)
            
            logger.info(f"Monitoring cycle completed. Metrics: {len(all_metrics)}, Alerts: {len(new_alerts)}")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
    
    async def run_continuous_monitoring(self, interval: int = 60):
        """Run continuous monitoring loop"""
        logger.info(f"Starting continuous monitoring with {interval}s interval")
        
        # Initialize database tables
        await self.initialize_database_tables()
        
        self.running = True
        
        try:
            while self.running:
                await self.run_monitoring_cycle()
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping monitoring system...")
        except Exception as e:
            logger.error(f"Critical error in monitoring system: {str(e)}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop monitoring system"""
        self.running = False
    
    async def get_system_dashboard_data(self) -> Dict[str, Any]:
        """Get current system dashboard data"""
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Get latest metrics
            latest_metrics = await conn.fetch("""
                SELECT DISTINCT ON (name) name, value, unit, timestamp, tags
                FROM system_metrics 
                ORDER BY name, timestamp DESC
            """)
            
            # Get active alerts
            active_alerts = await conn.fetch("""
                SELECT * FROM system_alerts 
                WHERE status = 'active' 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            # Get metrics summary
            metrics_summary = await conn.fetch("""
                SELECT 
                    tags->>'type' as category,
                    COUNT(*) as metric_count,
                    MAX(timestamp) as last_update
                FROM system_metrics 
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                GROUP BY tags->>'type'
            """)
            
            await conn.close()
            
            dashboard_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'healthy' if not active_alerts else 'alerts',
                'metrics': {
                    'latest': [dict(row) for row in latest_metrics],
                    'summary': [dict(row) for row in metrics_summary]
                },
                'alerts': {
                    'active_count': len(active_alerts),
                    'recent': [dict(row) for row in active_alerts]
                },
                'system': {
                    'monitoring_active': self.running,
                    'last_cycle': self.system_monitor.last_collection.isoformat() if self.system_monitor.last_collection else None
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'error',
                'error': str(e)
            }

async def main():
    """Main monitoring system execution"""
    # Configuration from environment
    db_url = os.getenv('DATABASE_URL', 'postgresql://pdm_user:password@localhost:5432/pdm_platform')
    api_endpoint = os.getenv('API_ENDPOINT', 'http://localhost:3000')
    monitoring_interval = int(os.getenv('MONITORING_INTERVAL', 60))
    
    # Initialize monitoring system
    monitor = ProductionMonitoringSystem(db_url, api_endpoint)
    
    # Start continuous monitoring
    await monitor.run_continuous_monitoring(monitoring_interval)

if __name__ == "__main__":
    asyncio.run(main())
