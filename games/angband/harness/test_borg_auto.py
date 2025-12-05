#!/usr/bin/env python3
"""
Automated borg test using pexpect.
Runs angband, creates character, starts borg, captures output.
"""

import pexpect
import time
import sys
import os
import re

ANGBAND_DIR = "/home/jw/dev/game1/external/angband-arena/angband/build"
ANGBAND_PATH = f"{ANGBAND_DIR}/game/angband"

def run_borg_test(duration_seconds=15):
    """Run borg for specified duration and capture results."""

    save_name = f"autotest_{int(time.time())}"

    # Change to build directory so relative paths work
    os.chdir(ANGBAND_DIR)
    print(f"Working dir: {os.getcwd()}")
    print(f"Starting Angband with save: {save_name}")
    print(f"Will run borg for {duration_seconds} seconds")
    print("-" * 50)

    # Start angband with new character
    child = pexpect.spawn(
        f'{ANGBAND_PATH} -mgcu -u"{save_name}" -n',
        encoding='utf-8',
        timeout=30,
        dimensions=(24, 80)
    )

    output_buffer = []

    def log_output(s):
        # Strip ANSI codes for cleaner output
        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', s)
        clean = re.sub(r'\x1b\][^\x07]*\x07?', '', clean)  # OSC sequences
        clean = re.sub(r'[\x00-\x1f]', '', clean)  # Control chars
        if clean.strip():
            output_buffer.append(clean)
            # Print last part
            if len(clean) > 5:
                print(f"[game] {clean[-80:]}")

    child.logfile_read = None  # Disable automatic logging

    try:
        # Wait for game to initialize
        time.sleep(2)

        # Read initial output
        try:
            child.expect(pexpect.TIMEOUT, timeout=1)
            log_output(child.before or "")
        except:
            pass

        print("\n--- Navigating character creation ---")

        # Character creation sequence:
        # 1. Press Enter to continue past news
        # 2. Select race (a = Human is usually first/default)
        # 3. Select class (a = Warrior usually)
        # 4. Accept stats (Enter/Return)
        # 5. Accept name (Enter)
        # 6. Accept (Enter)

        # Send 'a' for first race (Human)
        child.send('a')
        time.sleep(0.3)

        # Send 'a' for first class (Warrior)
        child.send('a')
        time.sleep(0.3)

        # Accept stats - Enter
        child.send('\r')
        time.sleep(0.3)

        # Accept name - Enter
        child.send('\r')
        time.sleep(0.3)

        # Final accept - Enter
        child.send('\r')
        time.sleep(0.5)

        # A few more enters to get through any remaining prompts
        for _ in range(5):
            child.send('\r')
            time.sleep(0.2)

        # Read output
        try:
            child.expect(pexpect.TIMEOUT, timeout=1)
            log_output(child.before or "")
        except:
            pass

        print("\n--- Starting borg (Ctrl-Z, z) ---")

        # Send Ctrl-Z to open borg menu
        child.send('\x1a')
        time.sleep(0.5)

        # Send 'z' to activate borg
        child.send('z')
        time.sleep(0.5)

        print(f"\n--- Borg running for {duration_seconds}s ---\n")

        # Let borg run and capture output
        start_time = time.time()
        last_output = ""
        while time.time() - start_time < duration_seconds:
            try:
                child.expect(pexpect.TIMEOUT, timeout=0.5)
                new_output = child.before or ""
                if new_output != last_output:
                    log_output(new_output)
                    last_output = new_output
            except:
                pass

        print("\n--- Stopping borg (space) ---")

        # Press space to stop borg
        child.send(' ')
        time.sleep(0.5)

        # Capture final state
        try:
            child.expect(pexpect.TIMEOUT, timeout=0.5)
            log_output(child.before or "")
        except:
            pass

        # Create log snapshot via borg menu
        child.send('\x1a')  # Ctrl-Z
        time.sleep(0.3)
        child.send('l')  # Log snapshot
        time.sleep(1)

        print("\n--- Quitting game ---")

        # Quit sequence
        child.send('\x1b')  # ESC
        time.sleep(0.2)
        child.send('Q')  # Quit
        time.sleep(0.2)
        child.send('y')  # Yes
        time.sleep(0.5)

        # Force quit if needed
        child.send('\x03')  # Ctrl-C

    except pexpect.EOF:
        print("Game ended (EOF)")
    except pexpect.TIMEOUT:
        print("Timeout")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        child.close()

    # Check for log files
    archive_dir = os.path.expanduser("~/.angband/Angband/archive")
    if os.path.exists(archive_dir):
        logs = sorted(os.listdir(archive_dir))
        if logs:
            print(f"\nLog files in archive: {logs[-5:]}")
            # Read the latest log
            latest = os.path.join(archive_dir, logs[-1])
            print(f"\n--- Latest log ({logs[-1]}) ---")
            with open(latest, 'r') as f:
                content = f.read()
                # Find depth info
                for line in content.split('\n'):
                    if 'Depth' in line or 'depth' in line or 'Level' in line:
                        print(line)
        else:
            print("\nNo log files yet")
    else:
        print("\nNo archive directory")

    # Check save file
    save_dir = f"{ANGBAND_DIR}/lib/save"
    if os.path.exists(save_dir):
        saves = os.listdir(save_dir)
        print(f"\nSave files: {saves}")

    print("\n--- Output buffer (last 20 lines) ---")
    for line in output_buffer[-20:]:
        print(line[:100])

    return True


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_borg_test(duration)
