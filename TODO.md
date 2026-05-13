# TODO: Enrich Data Pipeline Enhancement

## Steps
1. [ ] Update `voice_menu.py` — read from `dataset/`, add fuzzy matching, validate name before calling main.py
2. [ ] Update `main.py` — add `validate_person_exists()`, pass mode to `extract_faces()`
3. [ ] Update `02_extract.py` — cap max_images when enriching so predata doesn't exceed `MAX_IMAGES_PER_PERSON`
4. [ ] Update `02_augment.py` — robust "module check" to only augment newly added images, return count of newly augmented base images
5. [ ] Update `03_train.py` — improve log messages for update vs rebuild
6. [ ] Test pipeline with `--enrich <name>`

