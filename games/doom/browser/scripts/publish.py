#!/usr/bin/env python3
"""
GitHub Pages Publisher for DOOM Arena

Takes ranked checkpoints and builds a deployable GitHub Pages site with:
- index.html: Top 10 highlights selector
- play.html: DOOM WASM player with cheats
- checkpoints/: Savegame files for each highlight
- assets/: doom.wasm, doom.js, doom1.wad
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Template for highlight cards in index.html
HIGHLIGHT_CARD_TEMPLATE = '''
            <div class="highlight">
                <div class="highlight-header">
                    <span class="highlight-rank">#{rank}</span>
                    <span class="highlight-title">{title}</span>
                    <span class="ai-badge">{ai_model}</span>
                </div>
                <div class="highlight-body">
                    <div class="highlight-stats">
                        <div class="highlight-stat">
                            <div class="highlight-stat-value">{health}</div>
                            <div class="highlight-stat-label">Health</div>
                        </div>
                        <div class="highlight-stat">
                            <div class="highlight-stat-value">{kills}</div>
                            <div class="highlight-stat-label">Kills</div>
                        </div>
                        <div class="highlight-stat">
                            <div class="highlight-stat-value">{map_name}</div>
                            <div class="highlight-stat-label">Map</div>
                        </div>
                    </div>
                    <p class="highlight-desc">
                        {description}
                    </p>
                    <a href="play.html?save={save_file}" class="play-btn">
                        ▶ Play from here
                    </a>
                </div>
            </div>
'''


def generate_title(reason: str, description: str) -> str:
    """Generate a catchy title from the moment's reason."""
    titles = {
        'near_death': 'Near-Death Escape',
        'multi_kill': 'Multi-Kill Rampage',
        'boss': 'Boss Encounter',
        'death': 'Dramatic Fall',
    }
    return titles.get(reason, 'Epic Moment')


def build_pages(
    ranked_moments: list,
    checkpoint_dir: Path,
    output_dir: Path,
    template_dir: Path,
    wasm_dir: Path,
    wad_file: Path,
):
    """Build the GitHub Pages site."""

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'assets').mkdir(exist_ok=True)

    # Copy WASM assets if they exist
    wasm_files = ['doom.wasm', 'doom.js']
    for wf in wasm_files:
        src = wasm_dir / wf
        if src.exists():
            shutil.copy(src, output_dir / 'assets' / wf)
            print(f"Copied {wf}")
        else:
            print(f"Warning: {src} not found - build doom-wasm first")

    # Copy WAD file
    if wad_file.exists():
        shutil.copy(wad_file, output_dir / 'assets' / 'doom1.wad')
        print(f"Copied doom1.wad")
    else:
        print(f"Warning: {wad_file} not found")

    # Copy checkpoints and build highlight cards
    cards_html = []
    for i, moment in enumerate(ranked_moments[:10]):
        # Copy checkpoint file
        src_checkpoint = checkpoint_dir / moment['checkpoint']
        if src_checkpoint.exists():
            # Rename to simple format
            save_name = f"{i+1:03d}-{moment['reason']}"
            dst_checkpoint = output_dir / 'checkpoints' / f"{save_name}.sav"
            shutil.copy(src_checkpoint, dst_checkpoint)
        else:
            save_name = f"{i+1:03d}-placeholder"
            print(f"Warning: Checkpoint {src_checkpoint} not found")

        # Generate card HTML
        card = HIGHLIGHT_CARD_TEMPLATE.format(
            rank=i + 1,
            title=generate_title(moment['reason'], moment['description']),
            ai_model='GPU-AI',  # Could be model name from metadata
            health=moment['health'],
            kills=moment['kills'],
            map_name='E1M1',  # Could extract from checkpoint
            description=moment['description'],
            save_file=save_name,
        )
        cards_html.append(card)

    # Read and update index.html template
    index_template = template_dir / 'index.html'
    if index_template.exists():
        index_content = index_template.read_text()
        # Find placeholder and insert cards
        # The template has <!-- Populated by JavaScript or static -->
        placeholder = '<!-- Populated by JavaScript or static -->'
        if placeholder in index_content:
            # Remove placeholder and example cards, insert generated ones
            # Find the highlights div and replace content
            highlights_html = '\n'.join(cards_html)
            # Simple approach: replace placeholder with cards
            index_content = index_content.replace(
                placeholder,
                highlights_html + '\n' + placeholder
            )
        (output_dir / 'index.html').write_text(index_content)
        print("Generated index.html")
    else:
        print(f"Warning: Template not found: {index_template}")

    # Copy play.html
    play_template = template_dir / 'play.html'
    if play_template.exists():
        shutil.copy(play_template, output_dir / 'play.html')
        print("Copied play.html")

    # Generate manifest for debugging/CI
    manifest = {
        'generated': datetime.now().isoformat(),
        'moments': ranked_moments[:10],
        'assets': {
            'wasm': (wasm_dir / 'doom.wasm').exists(),
            'js': (wasm_dir / 'doom.js').exists(),
            'wad': wad_file.exists(),
        },
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print("Generated manifest.json")

    print(f"\nGitHub Pages site built in: {output_dir}")
    print("To deploy: git add pages/ && git commit && git push")


def main():
    parser = argparse.ArgumentParser(
        description='Build GitHub Pages site for DOOM Arena'
    )
    parser.add_argument('--ranked', type=Path, required=True,
                        help='JSON file with ranked moments (from rank_checkpoints.py --json)')
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='Directory containing checkpoint .bin/.sav files')
    parser.add_argument('--output', type=Path, default=Path('dist'),
                        help='Output directory (default: dist)')
    parser.add_argument('--templates', type=Path,
                        help='Template directory (default: ../pages)')
    parser.add_argument('--wasm', type=Path,
                        help='WASM build directory (default: ../wasm/build)')
    parser.add_argument('--wad', type=Path,
                        help='WAD file to use (default: ../../wads/doom1.wad)')

    args = parser.parse_args()

    # Set defaults relative to script location
    script_dir = Path(__file__).parent
    default_templates = script_dir.parent / 'pages'
    default_wasm = script_dir.parent / 'wasm' / 'build'
    default_wad = script_dir.parent.parent / 'wads' / 'doom1.wad'

    template_dir = args.templates or default_templates
    wasm_dir = args.wasm or default_wasm
    wad_file = args.wad or default_wad

    # Load ranked moments
    if not args.ranked.exists():
        print(f"Error: Ranked moments file not found: {args.ranked}", file=sys.stderr)
        sys.exit(1)

    moments = json.loads(args.ranked.read_text())
    if not moments:
        print("Error: No moments in ranked file", file=sys.stderr)
        sys.exit(1)

    print(f"Building site with {len(moments)} moments...")
    build_pages(
        ranked_moments=moments,
        checkpoint_dir=args.checkpoints,
        output_dir=args.output,
        template_dir=template_dir,
        wasm_dir=wasm_dir,
        wad_file=wad_file,
    )


if __name__ == '__main__':
    main()
