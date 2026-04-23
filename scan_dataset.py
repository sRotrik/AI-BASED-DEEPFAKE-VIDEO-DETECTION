from pathlib import Path

root = Path('D:/SROTRIK/deepfake_detection/FaceForensics++')
print('Root exists:', root.exists())
print()

if not root.exists():
    print('ERROR: D:/FaceForensics++ does not exist!')
    print('Check your drive letter and folder name.')
else:
    print('Top-level contents:')
    for item in sorted(root.iterdir()):
        kind = 'DIR ' if item.is_dir() else 'FILE'
        print(f'  [{kind}] {item.name}')

    print()
    print('All subfolders (up to 3 levels deep):')
    for item in sorted(root.rglob('*')):
        if item.is_dir():
            depth = len(item.relative_to(root).parts)
            if depth <= 3:
                indent = '  ' * depth
                print(f'{indent}{item.name}/')

    print()
    mp4s = list(root.rglob('*.mp4'))
    avis  = list(root.rglob('*.avi'))
    print(f'Total .mp4 files found anywhere: {len(mp4s)}')
    print(f'Total .avi files found anywhere: {len(avis)}')
    print()
    if mp4s:
        print('First 15 .mp4 paths:')
        for p in mp4s[:15]:
            print(f'  {p.relative_to(root)}')
    elif avis:
        print('First 15 .avi paths:')
        for p in avis[:15]:
            print(f'  {p.relative_to(root)}')
    else:
        print('No video files found at all!')
        print()
        print('All files found (first 20):')
        all_files = list(root.rglob('*.*'))
        for f in all_files[:20]:
            print(f'  {f.relative_to(root)}')