from logger_config import logger
import os
import subprocess
import shutil
import psutil
from decouple import config

RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
RAMDISK_SIZE_IN_GB = config("RAMDISK_SIZE_IN_GB", default=1, cast=int)

def check_that_user_has_required_permissions_to_manage_ramdisks():
    try: # Try to run a harmless command with sudo to test if the user has password-less sudo permissions
        result = subprocess.run(["sudo", "ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "password" in result.stderr.lower():
            raise PermissionError("Password required for sudo")
        logger.info("User has sufficient permissions to manage RAM Disks.")
        return True
    except (PermissionError, subprocess.CalledProcessError) as e:
        logger.info("Sorry, current user does not have sufficient permissions to manage RAM Disks! Disabling RAM Disks for now...")
        logger.debug(f"Permission check error detail: {e}")
        return False
    
def setup_ramdisk():
    if os.environ.get("RAMDISK_SETUP_DONE") == "1":
        logger.info("RAM Disk setup already completed by another worker. Skipping.")
        return    
    cmd_check = f"sudo mount | grep {RAMDISK_PATH}" # Check if RAM disk already exists at the path
    result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    if RAMDISK_PATH in result:
        logger.info(f"RAM Disk already set up at {RAMDISK_PATH}. Skipping setup.")
        return
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    free_ram_gb = psutil.virtual_memory().free / (1024 ** 3)
    buffer_gb = 2  # buffer to ensure we don't use all the free RAM
    ramdisk_size_gb = max(min(RAMDISK_SIZE_IN_GB, free_ram_gb - buffer_gb), 0.1)
    ramdisk_size_mb = int(ramdisk_size_gb * 1024)
    ramdisk_size_str = f"{ramdisk_size_mb}M"
    logger.info(f"Total RAM: {total_ram_gb}G")
    logger.info(f"Free RAM: {free_ram_gb}G")
    logger.info(f"Calculated RAM Disk Size: {ramdisk_size_gb}G")
    if RAMDISK_SIZE_IN_GB > total_ram_gb:
        raise ValueError(f"Cannot allocate {RAMDISK_SIZE_IN_GB}G for RAM Disk. Total system RAM is {total_ram_gb:.2f}G.")
    logger.info("Setting up RAM Disk...")
    os.makedirs(RAMDISK_PATH, exist_ok=True)
    mount_command = ["sudo", "mount", "-t", "tmpfs", "-o", f"size={ramdisk_size_str}", "tmpfs", RAMDISK_PATH]
    subprocess.run(mount_command, check=True)
    logger.info(f"RAM Disk set up at {RAMDISK_PATH} with size {ramdisk_size_gb}G")
    os.environ["RAMDISK_SETUP_DONE"] = "1"

def copy_models_to_ramdisk(models_directory, ramdisk_directory):
    total_size = sum(os.path.getsize(os.path.join(models_directory, model)) for model in os.listdir(models_directory))
    free_ram = psutil.virtual_memory().free
    if total_size > free_ram:
        logger.warning(f"Not enough space on RAM Disk. Required: {total_size}, Available: {free_ram}. Rebuilding RAM Disk.")
        clear_ramdisk()
        free_ram = psutil.virtual_memory().free  # Recompute the available RAM after clearing the RAM disk
        if total_size > free_ram:
            logger.error(f"Still not enough space on RAM Disk even after clearing. Required: {total_size}, Available: {free_ram}.")
            raise ValueError("Not enough RAM space to copy models.")
    os.makedirs(ramdisk_directory, exist_ok=True)
    for model in os.listdir(models_directory):
        src_path = os.path.join(models_directory, model)
        dest_path = os.path.join(ramdisk_directory, model)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == os.path.getsize(src_path): # Check if the file already exists in the RAM disk and has the same size
            logger.info(f"Model {model} already exists in RAM Disk and is the same size. Skipping copy.")
            continue
        shutil.copyfile(src_path, dest_path)
        logger.info(f"Copied model {model} to RAM Disk at {dest_path}")

def clear_ramdisk():
    while True:
        cmd_check = f"sudo mount | grep {RAMDISK_PATH}"
        result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        if RAMDISK_PATH not in result:
            break  # Exit the loop if the RAMDISK_PATH is not in the mount list
        cmd_umount = f"sudo umount -l {RAMDISK_PATH}"
        subprocess.run(cmd_umount, shell=True, check=True)
    logger.info(f"Cleared RAM Disk at {RAMDISK_PATH}")