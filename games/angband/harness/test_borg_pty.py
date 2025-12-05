#!/usr/bin/env python3
"""
Automated borg test using pty for proper terminal emulation.
"""

import pty
import os
import sys
import time
import select
import re
import subprocess

ANGBAND_DIR = "/home/jw/dev/game1/external/angband-arena/angband/build"
ANGBAND_BIN = f"{ANGBAND_DIR}/game/angband"

def strip_ansi(text):
    """Remove ANSI escape sequences."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07?|\x1b\(B')
    return ansi_escape.sub('', text)


def run_borg_test(duration=15):
    """Run angband with borg using PTY."""

    save_name = f"borgtest_{int(time.time())}"
    os.chdir(ANGBAND_DIR)

    print(f"=== Angband Borg Test ===")
    print(f"Save: {save_name}")
    print(f"Duration: {duration}s")
    print("=" * 40)

    # Create PTY
    master_fd, slave_fd = pty.openpty()

    # Set terminal size
    import fcntl
    import struct
    import termios
    winsize = struct.pack('HHHH', 24, 80, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    # Start angband
    cmd = [ANGBAND_BIN, '-mgcu', f'-u{save_name}', '-n']
    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        preexec_fn=os.setsid
    )

    os.close(slave_fd)

    output_buffer = []

    def write(data):
        """Write to the PTY."""
        os.write(master_fd, data.encode() if isinstance(data, str) else data)

    def read_available(timeout=0.5):
        """Read available output."""
        output = ""
        while True:
            r, _, _ = select.select([master_fd], [], [], timeout)
            if not r:
                break
            try:
                chunk = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                output += chunk
                timeout = 0.1  # Quick read after first chunk
            except OSError:
                break
        return output

    try:
        # Wait for game to start
        print("\n[Phase 1] Waiting for game to initialize...")
        time.sleep(2)
        initial = read_available(1)
        clean = strip_ansi(initial)
        if clean.strip():
            print(f"  Initial output: {len(clean)} chars")
            output_buffer.append(clean)

        # Navigate character creation
        print("\n[Phase 2] Character creation...")

        # Press space to continue past news
        write(' ')
        time.sleep(0.5)
        read_available(0.2)

        # Race: Human (a)
        print("  Selecting race: Human (a)")
        write('a')
        time.sleep(0.3)

        # Class: Warrior (a)
        print("  Selecting class: Warrior (a)")
        write('a')
        time.sleep(0.3)

        # Accept roller
        write('\r')
        time.sleep(0.3)

        # Name
        write('\r')
        time.sleep(0.3)

        # Final confirmation
        write('\r')
        time.sleep(0.5)

        # Extra confirmations
        for _ in range(3):
            write('\r')
            time.sleep(0.2)

        out = read_available(0.5)
        clean = strip_ansi(out)
        output_buffer.append(clean)
        print(f"  Character creation output: {len(clean)} chars")

        # Check if we're in game (look for @ symbol or similar)
        if '@' in clean or 'Warrior' in clean or 'Human' in clean:
            print("  Character appears to be created!")

        # Start borg
        print("\n[Phase 3] Starting borg (Ctrl-Z, z)...")
        write('\x1a')  # Ctrl-Z
        time.sleep(0.5)

        out = read_available(0.3)
        clean = strip_ansi(out)
        print(f"  After Ctrl-Z: {clean[:100]}..." if clean else "  (no output)")
        output_buffer.append(clean)

        # There may be a warning message - press Enter/space to continue through it
        write(' ')
        time.sleep(0.3)
        write('\r')
        time.sleep(0.3)

        out = read_available(0.3)
        clean = strip_ansi(out)
        if 'Borg' in clean or 'borg' in clean or 'z)' in clean:
            print("  Borg menu detected!")
        output_buffer.append(clean)

        write('z')  # Activate borg
        time.sleep(0.5)

        out = read_available(0.3)
        clean = strip_ansi(out)
        print(f"  After 'z': {clean[:100]}..." if clean else "  (no output)")

        out = read_available(0.3)
        clean = strip_ansi(out)
        output_buffer.append(clean)

        # Run borg
        print(f"\n[Phase 4] Borg running for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration:
            out = read_available(0.5)
            if out:
                clean = strip_ansi(out)
                output_buffer.append(clean)
                # Look for interesting events
                if 'die' in clean.lower() or 'kill' in clean.lower():
                    print(f"  Event: {clean[:60]}...")
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"  {int(elapsed)}s elapsed...")
            time.sleep(0.1)

        print(f"\n[Phase 5] Stopping borg...")
        write(' ')  # Stop borg
        time.sleep(0.5)

        # Get final output
        out = read_available(0.5)
        clean = strip_ansi(out)
        output_buffer.append(clean)

        # Create log snapshot
        print("  Creating log snapshot...")
        write('\x1a')  # Ctrl-Z
        time.sleep(0.3)
        write('l')  # Log
        time.sleep(1)

        out = read_available(0.5)
        output_buffer.append(strip_ansi(out))

        # Quit
        print("\n[Phase 6] Quitting...")
        write('\x1b')  # ESC
        time.sleep(0.2)
        write('Q')
        time.sleep(0.2)
        write('y')
        time.sleep(0.5)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except:
            proc.kill()
        os.close(master_fd)

    # Check results
    print("\n" + "=" * 40)
    print("=== Results ===")

    # Check archive for logs
    archive_dir = os.path.expanduser("~/.angband/Angband/archive")
    if os.path.exists(archive_dir):
        logs = sorted(os.listdir(archive_dir))
        if logs:
            print(f"\nLog files: {logs[-3:]}")
            latest = os.path.join(archive_dir, logs[-1])
            with open(latest, 'r') as f:
                content = f.read()
                # Extract key info
                for line in content.split('\n'):
                    if any(x in line for x in ['Depth', 'Level', 'Turns', 'Gold', 'HP']):
                        print(f"  {line.strip()}")

    # Check save files
    save_dir = f"{ANGBAND_DIR}/lib/save"
    if os.path.exists(save_dir):
        saves = os.listdir(save_dir)
        if saves:
            print(f"\nSave files: {saves}")

    # Show captured output summary
    full_output = ''.join(output_buffer)
    print(f"\nTotal output captured: {len(full_output)} chars")

    # Look for depth/level info in output
    depth_match = re.search(r'Depth:?\s*(\d+)', full_output)
    level_match = re.search(r'Level:?\s*(\d+)', full_output)
    if depth_match:
        print(f"Depth reached: {depth_match.group(1)}")
    if level_match:
        print(f"Character level: {level_match.group(1)}")

    # Show last bit of output
    if full_output:
        print("\n--- Last 500 chars of output ---")
        print(full_output[-500:])

    return True


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_borg_test(duration)
