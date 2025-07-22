from pathlib import Path
import random, shutil

SEED        = 42
SPLIT       = {'train': 0.70, 'val': 0.15, 'test': 0.15}
DATA_ROOT   = Path('datasets')           # folderele originale
OUT_ROOT    = Path('datasets/split_solarpanel')     # se generează

def main():
    random.seed(SEED)
    if OUT_ROOT.exists():
        print(f'{OUT_ROOT} există deja, șterge-l manual dacă vrei reințializare.'); return

    classes = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    for split in SPLIT:                     # creează arborele de directoare
        for cls in classes:
            Path(OUT_ROOT/split/cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        imgs = list((DATA_ROOT/cls).glob('*'))
        random.shuffle(imgs)
        n_total = len(imgs)
        n_train = int(SPLIT['train']*n_total)
        n_val   = int(SPLIT['val']  *n_total)

        split_map = (
            ('train', imgs[:n_train]),
            ('val',   imgs[n_train:n_train+n_val]),
            ('test',  imgs[n_train+n_val:])
        )
        for split, files in split_map:
            for f in files:
                dst = OUT_ROOT/split/cls/f.name
                try: dst.symlink_to(f.resolve())    # rapid & fără spațiu
                except OSError: shutil.copy2(f, dst) # pe Win fără admin

    print('♥  Split gata:', {k: round(v*100) for k,v in SPLIT.items()})

if __name__ == '__main__':
    main()
