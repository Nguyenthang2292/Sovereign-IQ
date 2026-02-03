"""
Secure Credential Manager
Handles secure storage and retrieval of API credentials using environment variables
"""
import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv, set_key, find_dotenv


class CredentialManager:
    """Manages secure storage of API credentials"""

    def __init__(self):
        """Initialize credential manager"""
        self.env_file = self._find_or_create_env_file()
        load_dotenv(self.env_file)

    def _find_or_create_env_file(self) -> Path:
        """Find or create .env file in project root"""
        # Try to find existing .env file
        env_path = find_dotenv()

        if env_path:
            return Path(env_path)

        # Create new .env file in project root
        # Navigate up from gui/utils/ to project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        env_file = project_root / ".env"

        if not env_file.exists():
            env_file.touch()
            # Add .env to .gitignore if not already there
            self._add_to_gitignore(project_root / ".gitignore")

        return env_file

    def _add_to_gitignore(self, gitignore_path: Path):
        """Ensure .env is in .gitignore"""
        try:
            if gitignore_path.exists():
                content = gitignore_path.read_text()
                if ".env" not in content:
                    with gitignore_path.open("a") as f:
                        f.write("\n# Environment variables\n.env\n")
            else:
                gitignore_path.write_text("# Environment variables\n.env\n")
        except Exception as e:
            print(f"Warning: Could not update .gitignore: {e}")

    def save_credentials(self, exchange: str, api_key: str, api_secret: str) -> bool:
        """
        Save API credentials securely to .env file

        Args:
            exchange: Exchange name (e.g., "binance", "demo")
            api_key: API key
            api_secret: API secret

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use exchange-specific environment variable names
            key_var = f"{exchange.upper()}_API_KEY"
            secret_var = f"{exchange.upper()}_API_SECRET"

            # Save to .env file
            set_key(str(self.env_file), key_var, api_key)
            set_key(str(self.env_file), secret_var, api_secret)

            # Reload environment variables
            load_dotenv(self.env_file, override=True)

            return True
        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False

    def load_credentials(self, exchange: str) -> Dict[str, Optional[str]]:
        """
        Load API credentials from environment variables

        Args:
            exchange: Exchange name (e.g., "binance", "demo")

        Returns:
            Dictionary with 'api_key' and 'api_secret' (may be None if not set)
        """
        key_var = f"{exchange.upper()}_API_KEY"
        secret_var = f"{exchange.upper()}_API_SECRET"

        return {
            "api_key": os.getenv(key_var),
            "api_secret": os.getenv(secret_var),
        }

    def has_credentials(self, exchange: str) -> bool:
        """
        Check if credentials exist for an exchange

        Args:
            exchange: Exchange name

        Returns:
            True if both API key and secret are set
        """
        creds = self.load_credentials(exchange)
        return bool(creds["api_key"] and creds["api_secret"])

    def clear_credentials(self, exchange: str) -> bool:
        """
        Clear credentials for an exchange

        Args:
            exchange: Exchange name

        Returns:
            True if successful
        """
        try:
            # Set empty values
            key_var = f"{exchange.upper()}_API_KEY"
            secret_var = f"{exchange.upper()}_API_SECRET"

            set_key(str(self.env_file), key_var, "")
            set_key(str(self.env_file), secret_var, "")

            # Reload
            load_dotenv(self.env_file, override=True)

            return True
        except Exception as e:
            print(f"Error clearing credentials: {e}")
            return False

    def test_connection(self, exchange: str, api_key: str, api_secret: str) -> Dict:
        """
        Test API connection with provided credentials

        Args:
            exchange: Exchange name
            api_key: API key to test
            api_secret: API secret to test

        Returns:
            Dictionary with 'success' (bool) and 'message' (str)
        """
        try:
            # Import here to avoid circular imports
            import ccxt

            # Map exchange names to ccxt exchange classes
            exchange_map = {
                "binance": ccxt.binance,
                "demo": ccxt.binance,  # Demo uses binance testnet
            }

            exchange_class = exchange_map.get(exchange.lower())
            if not exchange_class:
                return {
                    "success": False,
                    "message": f"Unsupported exchange: {exchange}"
                }

            # Initialize exchange with credentials
            exchange_instance = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            })

            # For demo/testnet
            if exchange.lower() == "demo":
                exchange_instance.set_sandbox_mode(True)

            # Test connection by fetching balance
            balance = exchange_instance.fetch_balance()

            # If we get here, connection is successful
            return {
                "success": True,
                "message": f"Successfully connected to {exchange}!",
                "balance": balance.get("total", {})
            }

        except ccxt.AuthenticationError as e:
            return {
                "success": False,
                "message": f"Authentication failed: Invalid API credentials"
            }
        except ccxt.NetworkError as e:
            return {
                "success": False,
                "message": f"Network error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}"
            }
