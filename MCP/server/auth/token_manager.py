"""
Token management for authentication
"""

import jwt
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from .models import User, TokenPayload


class TokenManager:
    """Manages JWT tokens and API key encryption"""

    _FERNNET_SALT = b"simplemem-fernet-v1"
    _PBKDF2_ITERATIONS = 480_000

    def __init__(
        self,
        secret_key: str,
        encryption_key: str,
        algorithm: str = "HS256",
        expiration_days: int = 30,
    ):
        if not secret_key or len(secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_days = expiration_days

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._FERNNET_SALT,
            iterations=self._PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
        self.fernet = Fernet(key)

    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt an API key for storage"""
        return self.fernet.encrypt(api_key.encode()).decode()

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt a stored API key"""
        return self.fernet.decrypt(encrypted_key.encode()).decode()

    def generate_token(self, user: User) -> str:
        """Generate a JWT token for a user"""
        expiration = datetime.now(timezone.utc) + timedelta(days=self.expiration_days)

        payload = TokenPayload(
            user_id=user.user_id,
            table_name=user.table_name,
            created_at=user.created_at.isoformat(),
            exp=int(expiration.timestamp()),
            token_version=int(user.token_version),
        )

        return jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm,
        )

    def verify_token(
        self, token: str
    ) -> Tuple[bool, Optional[TokenPayload], Optional[str]]:
        """
        Verify a JWT token

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        try:
            decoded = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            payload = TokenPayload.from_dict(decoded)
            return True, payload, None

        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"

    def refresh_token(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Refresh a valid token with new expiration

        Returns:
            Tuple of (new_token, error_message)
        """
        is_valid, payload, error = self.verify_token(token)
        if not is_valid:
            return None, error

        # Create new token with same user info but new expiration
        expiration = datetime.now(timezone.utc) + timedelta(days=self.expiration_days)

        new_payload = TokenPayload(
            user_id=payload.user_id,
            table_name=payload.table_name,
            created_at=payload.created_at,
            exp=int(expiration.timestamp()),
            token_version=int(payload.token_version),
        )

        new_token = jwt.encode(
            new_payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm,
        )

        return new_token, None
