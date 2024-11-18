'''
encryption.py -- Encrypt or decrypt a directory or file using Fernet symmetric encryption.

Usage: 
- python encryption.py generate-key /path/to/key (this will prompt for a passphrase which will be used to generate the key)
- python encryption.py process <encrypt|decrypt> /path/to/input /path/to/output -k /path/to/key (if no key is provided, the script will prompt for a passphrase)

Ex:
With current directory at project root:
- python3.10 ./scripts/encryption.py generate-key                                               './secrets/embedding_translation.fernet.key'
- python3.10 ./scripts/encryption.py process encrypt ./secrets           ./secrets.encrypted -k './secrets/embedding_translation.fernet.key'
- python3.10 ./scripts/encryption.py process decrypt ./secrets.encrypted ./secrets.decrypted -k './secrets/embedding_translation.fernet.key'
'''

import click
import pathlib
import tarfile
import io
from typing import Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
# pdkdf2 = Password-Based Key Derivation Function 2
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

def derive_key(passphrase: str) -> bytes:
    """
    Derive a Fernet key from a passphrase.

    Args:
        passphrase (str): The passphrase to derive the key from.

    Returns:
        bytes: The derived key.
    """
    # In a real scenario, use a random salt and store it securely
    salt = b'salt_'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key

def get_key(key_file: Union[str, None]) -> bytes:
    """
    Get the key from a file or prompt the user for a passphrase.

    Args:
        key_file (Union[str, None]): Path to the key file, or None if using a passphrase.

    Returns:
        bytes: The encryption/decryption key.

    Raises:
        click.ClickException: If there's an error reading the key file.
    """
    if key_file:
        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except IOError as e:
            raise click.ClickException(f"Error reading key file: {e}")
    else:
        passphrase = click.prompt('Enter the passphrase', hide_input=True, confirmation_prompt=True)
        return derive_key(passphrase)

def save_key(key: bytes, key_file: str):
    """
    Save the key to a file.

    Args:
        key (bytes): The key to save.
        key_file (str): Path to save the key file.

    Raises:
        click.ClickException: If there's an error writing the key file.
    """
    try:
        with open(key_file, 'wb') as f:
            f.write(key)
        click.echo(f"Key saved to {key_file}")
    except IOError as e:
        raise click.ClickException(f"Error saving key file: {e}")

def process_file(input_path: pathlib.Path, output_path: pathlib.Path, fernet: Fernet, encrypt: bool = True):
    """
    Process (encrypt or decrypt) a single file.

    Args:
        input_path (pathlib.Path): Path to the input file.
        output_path (pathlib.Path): Path to the output file.
        fernet (Fernet): The Fernet instance for encryption/decryption.
        encrypt (bool): True for encryption, False for decryption.

    Raises:
        click.ClickException: If there's an error reading, writing, or processing the file.
    """
    try:
        with input_path.open('rb') as f:
            data = f.read()
        
        if encrypt:
            processed_data = fernet.encrypt(data)
        else:
            processed_data = fernet.decrypt(data)
        
        with output_path.open('wb') as f:
            f.write(processed_data)
    except IOError as e:
        raise click.ClickException(f"Error processing file {input_path}: {e}")
    except InvalidToken:
        raise click.ClickException(f"Error decrypting file {input_path}: Invalid token. Ensure you're using the correct key.")

import tarfile
import io

def process_directory(input_path: pathlib.Path, output_path: pathlib.Path, fernet: Fernet, encrypt: bool = True):
    """
    Process (encrypt or decrypt) a directory into/from a single file.

    Args:
        input_path (pathlib.Path): Path to the input directory (for encryption) or file (for decryption).
        output_path (pathlib.Path): Path to the output file (for encryption) or directory (for decryption).
        fernet (Fernet): The Fernet instance for encryption/decryption.
        encrypt (bool): True for encryption, False for decryption.

    Raises:
        click.ClickException: If there's an error processing the directory.
    """
    try:
        if encrypt:
            # Create a tar archive in memory
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
                tar.add(input_path, arcname='.')
            
            # Encrypt the tar archive
            encrypted_data = fernet.encrypt(tar_buffer.getvalue())
            
            # Write the encrypted data to the output file
            with output_path.open('wb') as f:
                f.write(encrypted_data)
        else:
            # Read the encrypted file
            with input_path.open('rb') as f:
                encrypted_data = f.read()
            
            # Decrypt the data
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Check if the decrypted data is a tar file
            tar_buffer = io.BytesIO(decrypted_data)
            if tarfile.is_tarfile(tar_buffer):
                # Extract the tar archive to the output directory
                tar_buffer.seek(0)  # Reset buffer position
                with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
                    tar.extractall(path=output_path)
            else:
                # If it's not a tar file, write the decrypted data directly
                with output_path.open('wb') as f:
                    f.write(decrypted_data)
        
        click.echo(f"Successfully {'encrypted' if encrypt else 'decrypted'} {'directory' if encrypt else 'file or directory'}")
    except Exception as e:
        raise click.ClickException(f"Error processing {'directory' if encrypt else 'file or directory'}: {e}")

@click.group()
def cli():
    """Encrypt, decrypt, or generate key for files and directories."""
    pass

@cli.command()
@click.argument('action', type=click.Choice(['encrypt', 'decrypt']))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('-k', '--key-file', type=click.Path(exists=True), help='Path to the key file')
def process(action: str, input_path: str, output_path: str, key_file: Union[str, None]):
    try:
        key = get_key(key_file)
        fernet = Fernet(key)

        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

        if input_path.is_file() and action == 'encrypt':
            process_file(input_path, output_path, fernet, True)
        elif input_path.is_dir() and action == 'encrypt':
            process_directory(input_path, output_path, fernet, True)
        elif input_path.is_file() and action == 'decrypt':
            process_directory(input_path, output_path, fernet, False)
        else:
            raise click.ClickException(f"Error: Invalid input/output combination for {action}")

        click.echo(f"Successfully {action}ed {input_path} to {output_path}")
    except click.ClickException as e:
        click.echo(str(e), err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

@cli.command()
@click.argument('key_file', type=click.Path())
def generate_key(key_file: str):
    """
    Generate a key file from a passphrase.

    Args:
        key_file (str): Path to save the generated key file.
    """
    try:
        passphrase = click.prompt('Enter the passphrase to generate the key', hide_input=True, confirmation_prompt=True)
        key = derive_key(passphrase)
        save_key(key, key_file)
        click.echo(f"Key generated and saved to {key_file}")
    except click.ClickException as e:
        click.echo(str(e), err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)

if __name__ == '__main__':
    cli()