import pytest
import os
import subprocess
from swiss_army_llama import check_that_user_has_required_permissions_to_manage_ramdisks, setup_ramdisk, copy_models_to_ramdisk, clear_ramdisk, RAMDISK_PATH

@pytest.mark.skipif(not os.environ.get('RUN_SUDO_TESTS'), reason="requires admin rights")
def test_check_user_permission_for_ramdisk():
    assert check_that_user_has_required_permissions_to_manage_ramdisks() is True

@pytest.mark.skipif(not os.environ.get('RUN_SUDO_TESTS'), reason="requires admin rights")
def test_setup_ramdisk():
    setup_ramdisk()
    cmd_check = f"mount | grep {RAMDISK_PATH}"
    result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    assert RAMDISK_PATH in result

@pytest.mark.skipif(not os.environ.get('RUN_SUDO_TESTS'), reason="requires admin rights")
def test_copy_models_to_ramdisk():
    models_directory = "./test_models/"
    ramdisk_directory = RAMDISK_PATH
    os.makedirs(models_directory, exist_ok=True)
    with open(f"{models_directory}/dummy_model.bin", "wb") as f:
        f.write(b"Dummy data")
    copy_models_to_ramdisk(models_directory, ramdisk_directory)
    assert os.path.exists(os.path.join(ramdisk_directory, "dummy_model.bin")) is True

@pytest.mark.skipif(not os.environ.get('RUN_SUDO_TESTS'), reason="requires admin rights")
def test_clear_ramdisk():
    clear_ramdisk()
    cmd_check = f"mount | grep {RAMDISK_PATH}"
    result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    assert RAMDISK_PATH not in result
