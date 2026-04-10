import re

def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(text).lower())).strip()

def normalize_for_comparison(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\b(phase|tower|block|wing)\s*[0-9ivx]+\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = {'the','by','at','in','on','near','apartment','apartments','villa','villas','project','residential','commercial'}
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def calculate_similarity(prop1, prop2):
    score = 0
    name1 = normalize_for_comparison(prop1.get('project_name') or prop1.get('name') or prop1.get('title', ''))
    name2 = normalize_for_comparison(prop2.get('project_name') or prop2.get('name') or prop2.get('title', ''))
    if name1 and name2:
        if name1 == name2:
            score += 40
        elif name1 in name2 or name2 in name1:
            score += 35
        else:
            w1, w2 = set(name1.split()), set(name2.split())
            if w1 and w2:
                score += int(40 * len(w1 & w2) / len(w1 | w2))
    city1 = normalize_text(prop1.get('city', ''))
    city2 = normalize_text(prop2.get('city', ''))
    if city1 and city2:
        if city1 == city2:
            score += 20
        elif city1 in city2 or city2 in city1:
            score += 15
    b1 = normalize_for_comparison(prop1.get('builder') or prop1.get('builder_name', ''))
    b2 = normalize_for_comparison(prop2.get('builder') or prop2.get('builder_name', ''))
    if b1 and b2 and b1 != 'unknown' and b2 != 'unknown':
        if b1 == b2:
            score += 20
        elif b1 in b2 or b2 in b1:
            score += 15
    l1 = normalize_for_comparison(prop1.get('location') or prop1.get('locality', ''))
    l2 = normalize_for_comparison(prop2.get('location') or prop2.get('locality', ''))
    if l1 and l2:
        if l1 == l2:
            score += 10
        elif l1 in l2 or l2 in l1:
            score += 7
    return min(score, 100)
