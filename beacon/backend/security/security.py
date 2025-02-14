"""
Security enhancement module for BEACON

This module implements data encryption, access control, and security checks.
"""

import jwt
import bcrypt
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os

class SecurityManager:
    """Security management class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize security manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption keys
        self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def encrypt_data(self, data: Any) -> bytes:
        """Encrypt data using Fernet (symmetric encryption).

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        try:
            serialized_data = self._serialize_data(data)
            return self.fernet.encrypt(serialized_data)
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt data using Fernet (symmetric encryption).

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data
        """
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return self._deserialize_data(decrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            raise

    def asymmetric_encrypt(self, data: bytes) -> bytes:
        """Encrypt data using RSA (asymmetric encryption).

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        try:
            return self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        except Exception as e:
            self.logger.error(f"Asymmetric encryption error: {str(e)}")
            raise

    def asymmetric_decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA (asymmetric encryption).

        Args:
            encrypted_data: Encrypted data

        Returns:
            Decrypted data
        """
        try:
            return self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        except Exception as e:
            self.logger.error(f"Asymmetric decryption error: {str(e)}")
            raise

    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key using PBKDF2.

        Returns:
            Generated key
        """
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(
            self.config['secret_key'].encode()
        ))
        return key

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for encryption.

        Args:
            data: Data to serialize

        Returns:
            Serialized data
        """
        if isinstance(data, str):
            return data.encode()
        elif isinstance(data, (dict, list)):
            return json.dumps(data).encode()
        else:
            return pickle.dumps(data)

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize decrypted data.

        Args:
            data: Data to deserialize

        Returns:
            Deserialized data
        """
        try:
            return json.loads(data)
        except:
            try:
                return pickle.loads(data)
            except:
                return data.decode()

class AccessControl:
    """Access control management class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize access control.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.secret_key = config['secret_key']
        
        # Define role permissions
        self.role_permissions = {
            'admin': {'read', 'write', 'delete', 'manage_users'},
            'doctor': {'read', 'write', 'limited_delete'},
            'nurse': {'read', 'limited_write'},
            'patient': {'read_own', 'write_own'}
        }

    def create_token(self, user_id: str, role: str) -> str:
        """Create JWT token.

        Args:
            user_id: User identifier
            role: User role

        Returns:
            JWT token
        """
        payload = {
            'user_id': user_id,
            'role': role,
            'permissions': list(self.role_permissions.get(role, [])),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token.

        Args:
            token: JWT token

        Returns:
            Token payload
        """
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def check_permission(self, token: str, required_permission: str,
                        resource_owner_id: Optional[str] = None) -> bool:
        """Check if user has required permission.

        Args:
            token: JWT token
            required_permission: Required permission
            resource_owner_id: Resource owner ID for own-resource permissions

        Returns:
            True if permitted, False otherwise
        """
        try:
            payload = self.verify_token(token)
            permissions = set(payload['permissions'])
            
            # Check for admin override
            if 'admin' in permissions:
                return True
            
            # Check own-resource permissions
            if required_permission.endswith('_own'):
                return (resource_owner_id == payload['user_id'] and
                       required_permission in permissions)
            
            return required_permission in permissions
            
        except Exception as e:
            self.logger.error(f"Permission check error: {str(e)}")
            return False

class SecurityAuditor:
    """Security auditing class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize security auditor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audit_file = Path(config.get('audit_file', 'security_audit.log'))

    def log_security_event(self, event_type: str, user_id: str,
                          details: Dict[str, Any]) -> None:
        """Log security event.

        Args:
            event_type: Type of security event
            user_id: User identifier
            details: Event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details
        }
        
        with open(self.audit_file, 'a') as f:
            json.dump(event, f)
            f.write('\n')

    def analyze_security_logs(self, time_window: timedelta) -> Dict[str, Any]:
        """Analyze security logs.

        Args:
            time_window: Time window for analysis

        Returns:
            Analysis results
        """
        events = []
        start_time = datetime.now() - time_window
        
        with open(self.audit_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                event_time = datetime.fromisoformat(event['timestamp'])
                if event_time >= start_time:
                    events.append(event)
        
        return self._analyze_events(events)

    def _analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security events.

        Args:
            events: List of security events

        Returns:
            Analysis results
        """
        analysis = {
            'total_events': len(events),
            'event_types': {},
            'user_activity': {},
            'suspicious_activity': []
        }
        
        for event in events:
            # Count event types
            event_type = event['event_type']
            analysis['event_types'][event_type] = analysis['event_types'].get(
                event_type, 0) + 1
            
            # Track user activity
            user_id = event['user_id']
            if user_id not in analysis['user_activity']:
                analysis['user_activity'][user_id] = []
            analysis['user_activity'][user_id].append(event)
            
            # Check for suspicious activity
            if self._is_suspicious(event):
                analysis['suspicious_activity'].append(event)
        
        return analysis

    def _is_suspicious(self, event: Dict[str, Any]) -> bool:
        """Check if event is suspicious.

        Args:
            event: Security event

        Returns:
            True if suspicious, False otherwise
        """
        # Implement suspicious activity detection logic
        return False 