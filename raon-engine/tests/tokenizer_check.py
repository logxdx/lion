#!/usr/bin/env python3
"""Token-id parity of build/raon-tok against KRAFTON llama.cpp's llama-tokenize.

    python3 tests/tokenizer_check.py --model M.gguf --ref ~/llama.cpp/raon-src/build/bin/llama-tokenize \
        [--files a.txt b.cu ...] [--fuzz 150]
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import unicodedata

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--raon', default=os.path.join(os.path.dirname(__file__), '..', 'build', 'raon-tok'))
ap.add_argument('--files', nargs='*', default=[])
ap.add_argument('--fuzz', type=int, default=150)
ap.add_argument('--seed', type=int, default=1)
args = ap.parse_args()


def ids(cmd):
    out = subprocess.run(cmd, capture_output=True, check=True).stdout.decode().strip().splitlines()
    line = [l for l in out if l.startswith('[')][-1]
    return [int(x) for x in line.strip('[]').split(',') if x.strip()]


def compare(name, data, special=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(data)
        path = f.name
    try:
        ref = ids([args.ref, '-m', args.model, '-f', path, '--ids', '--log-disable', '--no-escape'] + ([] if special else ['--no-parse-special']))
        got = ids([args.raon, '-m', args.model, '-f', path] + ([] if special else ['--no-special']))
    finally:
        os.unlink(path)
    if ref == got:
        return True
    k = next((i for i in range(min(len(ref), len(got))) if ref[i] != got[i]), min(len(ref), len(got)))
    print(f'MISMATCH {name}: ref {len(ref)} tokens, raon {len(got)}; first diff at {k}: ref {ref[k:k+6]} raon {got[k:k+6]}')
    print('   text around:', repr(data[:400]))
    return False


rng = random.Random(args.seed)
pools = {
    'ascii': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'digit': '0123456789',
    'space': [' ', ' ', ' ', '  ', '\t', '\n', '\r\n', '\n\n', '   \n', '  ', '　', ' '],
    'punct': list('.,;:!?\'"()[]{}<>/\\|-_=+*&^%$#@~`') + ['...', '://', '—', '«', '»', '“', '”', '・', '。', '、'],
    'mark': ['́', '̀', '̈', '̧', '̣', '̂', 'َ', 'ा', 'ั', '⃗'],
    'cjk': list('的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年'),
    'kana': list('あいうえおかきくけこさしすせそアイウエオカキクケコッャー'),
    'hangul': list('가나다라마바사아자차카타파하한국어입니다요') + ['가', '한'],
    'latin': list('éèêëàâäôöûüçñßøåæœÉÀÇÑ'),
    'cyr': list('абвгдежзийклмнопрстуфхцчшщыэюяАБВГД'),
    'greek': list('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔ'),
    'arabic': list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي'),
    'deva': list('अआइईउऊकखगघङचछजझटठडढणतथदधनपफबभमयरलवशषसह'),
    'emoji': ['😀', '👍🏽', '👨‍👩‍👧', '🇰🇷', '❤️', '🚀', '‍', '​', '﻿'],
    'num': ['１２３', '٣٤٥', '½', '²', 'Ⅻ', '१२'],
}
keys = list(pools)


def fuzz_text():
    parts = []
    for _ in range(rng.randint(5, 80)):
        k = rng.choice(keys)
        n = rng.randint(1, 8)
        parts.append(''.join(rng.choice(pools[k]) for _ in range(n)))
        if rng.random() < 0.5:
            parts.append(' ')
    return ''.join(parts)


cases = [
    ('words joined by single spaces', b'of the people by the people for the people shall not perish from the earth'),
    ('numbers', b'1 12 123 1234 12345 123456 1234567 3.14159 1,000,000 0x1F 2026-11-25 v1.2.3'),
    ('whitespace', b'a  b   c\t\td\n\ne \r\n f   \n   g    '),
    ('trailing spaces', b'end with spaces   '),
    ('leading spaces', b'   start'),
    ('only newlines', b'\n\n\n'),
    ('code', b'def f(x):\n    return x ** 2  # square\n\nfor i in range(10):\n\tprint(f"{i}: {f(i)}")\n'),
    ('url path', b'see https://example.com/a/b/c?x=1&y=2 and ./src/main.cpp:42 or /usr/bin/env'),
    ('specials', b'<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call><|im_end|>'),
    ('special-like', b'<|im_sta <think <|reserved_99|> <|reserved_9999|> <|fim_prefix|>x<|fim_suffix|>'),
    ('NFD input', unicodedata.normalize('NFD', 'café résumé naïve Ångström 한국어 ﬁ ｱ').encode()),
    ('invalid utf8', b'abc\xff\xfe def \xc3 ghi \xe2\x82 jkl \xf0\x9f\x98 mno \xed\xa0\x80'),
    ('korean', '대한민국의 수도는 서울입니다. 오늘 날씨가 정말 좋네요! 크래프톤은 게임 회사입니다.'.encode()),
    ('japanese', '東京は日本の首都です。カタカナとひらがなを混ぜて書きます。ラーメンが好き！'.encode()),
    ('chinese', '人工智能正在改变世界，我们需要负责任地发展它。'.encode()),
    ('mixed', 'The 한국어 word 日本語 and 中文 mixed with English. 価格は1,234円です。'.encode()),
    ('json', b'{"name": "raon", "values": [1, 2, 3], "nested": {"a": null, "b": true}}'),
    ('markdown', b'# Title\n\n- item one\n- item two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nx = 1\n```\n'),
    ('empty', b''),
    ('single space', b' '),
]
for p in args.files:
    cases.append((os.path.basename(p), open(p, 'rb').read()))
for i in range(args.fuzz):
    cases.append((f'fuzz{i}', fuzz_text().encode('utf-8', 'surrogatepass')))

ok = 0
for name, data in cases:
    ok += compare(name, data)
for name, data in cases[:12]:
    ok += compare(name + ' (no special)', data, special=False)
total = len(cases) + 12
print(f'{ok}/{total} cases token-identical')
sys.exit(0 if ok == total else 1)
