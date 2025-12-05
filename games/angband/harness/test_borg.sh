#!/bin/bash
# Test borg startup
# Run in tmux or with PTY

cd /home/jw/dev/game1/external/angband-arena/angband/build

# Create a test save file name
SAVENAME="arena_test_$(date +%s)"

echo "Starting Angband with borg support..."
echo "Save file: $SAVENAME"
echo ""
echo "Instructions:"
echo "1. Create a new character (any race/class)"
echo "2. Press ^z (Ctrl-Z) to open borg menu"
echo "3. Press z to start the borg"
echo "4. Watch it play!"
echo "5. Press any key to stop borg"
echo "6. Press q to quit angband"
echo ""
echo "Press Enter to start..."
read

./game/angband -mgcu -u"$SAVENAME" -n
