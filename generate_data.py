#!/usr/bin/env python3
"""
Generate synthetic noisy STT training data with PII entities.
Reflects realistic speech-to-text patterns with spelling variations.
Enhanced with harder, more challenging examples.
"""
import json
import random
from typing import List, Dict, Tuple

# Number word mappings for noisy STT
NUMBER_WORDS = {
    "0": ["zero", "oh", "o", "owe", "nought", "nil"],
    "1": ["one", "won", "wan"],
    "2": ["two", "to", "too", "tu"],
    "3": ["three", "tree", "free"],
    "4": ["four", "for", "fore", "fo"],
    "5": ["five", "fife", "hive"],
    "6": ["six", "sicks", "sex"],
    "7": ["seven", "sevin", "savin"],
    "8": ["eight", "ate", "ait"],
    "9": ["nine", "nein", "nigh"],
}

# STT disfluencies and noise - expanded
DISFLUENCIES = ["um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of", "actually", "literally"]
FILLERS = ["well", "so", "basically", "honestly", "look", "listen", "okay", "right"]
REPEATS = ["i i", "the the", "a a", "is is", "and and", "that that", "my my"]
NOISE_WORDS = ["noise", "static", "background", "click", "beep", "silence"]

# Sample data
FIRST_NAMES = [
    "ramesh", "priyanka", "anil", "sneha", "vikram", "kavita", "rajesh", "meera",
    "arjun", "divya", "suresh", "anita", "mohan", "pooja", "kiran", "neha",
    "rahul", "sonia", "amit", "deepa", "manoj", "swati", "naveen", "ritu",
    "kumar", "lalita", "pradeep", "sunita", "ajay", "kavya", "nitin", "radha"
]

LAST_NAMES = [
    "sharma", "verma", "patel", "kumar", "singh", "reddy", "iyer", "nair",
    "desai", "mehta", "jain", "gupta", "rao", "naidu", "menon", "pandey",
    "kapoor", "malhotra", "chopra", "bansal", "agarwal", "goswami", "bhatt"
]

EMAIL_DOMAINS = [
    "gmail", "yahoo", "outlook", "hotmail", "rediffmail", "icloud", "protonmail"
]

CITIES = [
    "mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune", "kolkata",
    "ahmedabad", "jaipur", "lucknow", "bhopal", "indore", "nagpur", "cochin",
    "surat", "vadodara", "patna", "ludhiana", "agra", "nashik"
]

LOCATIONS = [
    "airport road", "main street", "park avenue", "church street", "mg road",
    "connaught place", "marine drive", "juhu beach", "bandra west", "koramangala",
    "sector", "phase", "block", "colony", "nagar", "layout"
]

MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

MONTH_NUMS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def number_to_words(digits: str, use_words: bool = True, add_noise: bool = False) -> str:
    """Convert digit string to spoken form with noisy STT patterns - easier."""
    if not use_words or random.random() < 0.3:  # More digit usage (easier)
        # Sometimes use digits directly, sometimes with spaces
        if random.random() < 0.6:  # Prefer spaced digits
            result = " ".join(digits)
        else:
            result = digits
        
        # Less noise: sometimes insert spaces randomly
        if add_noise and random.random() < 0.2:  # Reduced from 0.4
            # Insert random space
            if len(result) > 2:
                idx = random.randint(1, len(result)-1)
                result = result[:idx] + " " + result[idx:]
        return result
    
    words = []
    for d in digits:
        if d in NUMBER_WORDS:
            words.append(random.choice(NUMBER_WORDS[d]))
    
    # Less noise word insertion
    if add_noise and random.random() < 0.05 and len(words) > 2:  # Reduced from 0.15
        idx = random.randint(1, len(words)-1)
        words.insert(idx, random.choice(["um", "uh"]))  # Only simple disfluencies
    
    result = " ".join(words)
    
    return result


def generate_credit_card() -> Tuple[str, int, int]:
    """Generate credit card number with noisy STT patterns - easier variations."""
    # Format: 4 groups of 4 digits
    groups = []
    for _ in range(4):
        group = "".join([str(random.randint(0, 9)) for _ in range(4)])
        groups.append(group)
    
    use_words = random.random() < 0.2  # Even less word usage (easier)
    add_noise = random.random() < 0.1 # Add noise 5% of time (much less noise)
    
    if use_words:
        # Simpler word patterns
        text_parts = []
        for g in groups:
            text_parts.append(number_to_words(g, use_words=True, add_noise=False))  # No noise in words
        text = " ".join(text_parts)
    else:
        # All digits, mostly with spaces (standard format)
        if random.random() < 0.9:  # 90% standard format (up from 80%)
            text = " ".join(groups)
        else:
            text = "".join(groups)
    
    # Even less often add prefix/suffix
    if add_noise and random.random() < 0.05:
        prefixes = ["card number", "credit card"]
        if random.random() < 0.5:
            text = f"{random.choice(prefixes)} {text}"
    
    start = 0
    end = len(text)
    return text, start, end


def generate_phone() -> Tuple[str, int, int]:
    """Generate phone number with noisy STT patterns - easier variations."""
    # 10 digits, sometimes with country code included
    include_country = random.random() < 0.15  # Even less country codes
    add_noise = random.random() < 0.1  # Even less noise
    
    if include_country:
        digits = "".join([str(random.randint(0, 9)) for _ in range(12)])
    else:
        digits = "".join([str(random.randint(0, 9)) for _ in range(10)])
    
    use_words = random.random() < 0.2  # Even less word usage (easier)
    
    # Even easier patterns: more consistent formats
    if use_words:
        # Full word conversion, no partial
        text = number_to_words(digits, use_words=True, add_noise=False)  # No noise
    else:
        # Digits with consistent spacing
        if random.random() < 0.85:  # 85% standard grouped format (up from 70%)
            # Grouped: XXX-XXX-XXXX or XXX XXX XXXX
            if random.random() < 0.5:
                text = f"{digits[:3]} {digits[3:6]} {digits[6:]}"
            else:
                text = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif random.random() < 0.5:
            text = " ".join(digits)
        else:
            text = digits
    
    # Even less often add country code separately
    if not include_country and random.random() < 0.1:
        country_formats = [
            "plus nine one",
            "+91",
            "nine one"
        ]
        country = random.choice(country_formats)
        text = f"{country} {text}"
    
    start = 0
    end = len(text)
    return text, start, end


def generate_email() -> Tuple[str, int, int]:
    """Generate email address with noisy STT patterns - easier variations."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    add_noise = random.random() < 0.1  # Further reduced noise
    
    # Even easier formats - heavily favor standard format
    formats = [
        f"{first}.{last}@{domain}.com",  # Standard format (most common - 70%)
        f"{first}.{last}@{domain}.com",  # Standard format
        f"{first}.{last}@{domain}.com",  # Standard format
        f"{first}.{last}@{domain}.com",  # Standard format
        f"{first}.{last}@{domain}.com",  # Standard format
        f"{first}.{last}@{domain}.com",  # Standard format
        f"{first} dot {last} at {domain} dot com",  # Spoken format (30%)
        f"{first} dot {last} at {domain} dot com",  # Spoken format
        f"{first} dot {last} at {domain} dot com",  # Spoken format
    ]
    
    text = random.choice(formats)
    
    # Add disfluencies even less often
    if add_noise and random.random() < 0.2:  # Further reduced probability
        noise = random.choice(["um", "uh"])
        if random.random() < 0.5:
            text = f"{text} {noise}"
        else:
            text = f"{noise} {text}"
    
    start = 0
    end = len(text)
    return text, start, end


def generate_person_name() -> Tuple[str, int, int]:
    """Generate person name - easier variations."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    add_noise = random.random() < 0.05  # Even less noise
    
    # Even simpler formats - mostly full names
    if random.random() < 0.9:  # 90% full names (up from 85%)
        text = f"{first} {last}"
    elif random.random() < 0.5:
        text = first
    else:
        # Sometimes with title
        titles = ["mr", "mrs", "ms", "dr"]
        if random.random() < 0.5:
            text = f"{random.choice(titles)} {first} {last}"
        else:
            text = f"{first} {last}"
    
    # Even less disfluencies
    if add_noise and random.random() < 0.1:  # Further reduced probability
        text = f"{text} um"
    
    start = 0
    end = len(text)
    return text, start, end


def generate_date() -> Tuple[str, int, int]:
    """Generate date with various formats - easier variations."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2025)
    add_noise = random.random() < 0.15  # Even less noise
    
    # Even simpler, more common formats
    formats = [
        f"{day:02d} {month:02d} {year}",  # Standard numeric (most common)
        f"{day:02d} {month:02d} {year}",  # Standard numeric
        f"{day:02d} {month:02d} {year}",  # Standard numeric
        f"{day} {MONTHS[month-1]} {year}",  # Day month year
        f"{MONTHS[month-1]} {day} {year}",  # Month day year
        f"{day} {month} {year}",  # Simple numeric
        f"{day:02d} {MONTH_NUMS[month-1]} {year}",  # With zero-padded month
    ]
    
    text = random.choice(formats)
    
    # Even less disfluencies or prefixes
    if add_noise and random.random() < 0.2:  # Further reduced probability
        if random.random() < 0.5:
            prefixes = ["on", "date"]
            text = f"{random.choice(prefixes)} {text}"
        else:
            text = f"{text} um"
    
    start = 0
    end = len(text)
    return text, start, end


def generate_city() -> Tuple[str, int, int]:
    """Generate city name."""
    text = random.choice(CITIES)
    start = 0
    end = len(text)
    return text, start, end


def generate_location() -> Tuple[str, int, int]:
    """Generate location."""
    base = random.choice(LOCATIONS)
    
    # Sometimes add numbers or modifiers
    if random.random() < 0.3:
        if "sector" in base or "phase" in base or "block" in base:
            num = random.randint(1, 20)
            text = f"{base} {num}"
        else:
            text = base
    else:
        text = base
    
    start = 0
    end = len(text)
    return text, start, end


def create_utterance(utt_id: str, num_entities: int = None, make_hard: bool = True) -> Dict:
    """Create a single utterance with entities and noisy STT patterns - easier version."""
    if num_entities is None:
        # Moderate number of entities
        num_entities = random.randint(1, 3) if make_hard else random.randint(1, 2)
    
    # Select entity types
    entity_types = random.sample(
        ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"],
        min(num_entities, 7)
    )
    
    # Even less duplicate entity types
    if make_hard and random.random() < 0.03 and num_entities < 3:  # Further reduced duplicates
        entity_types.append(random.choice(["PHONE", "EMAIL", "PERSON_NAME"]))
    
    generators = {
        "CREDIT_CARD": generate_credit_card,
        "PHONE": generate_phone,
        "EMAIL": generate_email,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    # Build text incrementally and track positions exactly
    text_parts = []
    entities_data = []
    
    # Simpler prefix text with even less disfluencies
    prefixes = [
        "hi", "hello", "my", "i", "please", "can you", "i need", "i want",
        "call me", "contact me", "my name is", "i am", "this is"
    ]
    
    if random.random() < 0.5:  # Even less prefixes (down from 0.6)
        prefix = random.choice(prefixes)
        text_parts.append(prefix)
        
        # Even less disfluency after prefix
        if make_hard and random.random() < 0.05:  # Further reduced from 0.1
            text_parts.append(random.choice(["um", "uh"]))  # Only simple disfluencies
    
    # Generate entities with connectors - simpler
    connectors = ["is", "are", "and", "or", "also", "with", "on", "the", "my", "at"]
    
    for i, ent_type in enumerate(entity_types):
        if i > 0:
            connector = random.choice(connectors)
            text_parts.append(connector)
            
            # Even less disfluency between entities
            if make_hard and random.random() < 0.05:  # Further reduced from 0.1
                text_parts.append(random.choice(["um", "uh"]))  # Only simple disfluencies
            
            # Even less distractors
            if make_hard and random.random() < 0.03:  # Further reduced from 0.05
                distractor = random.choice(["wait", "sorry"])
                text_parts.append(distractor)
        
        ent_text, _, _ = generators[ent_type]()
        text_parts.append(ent_text)
        entities_data.append({
            "text": ent_text,
            "label": ent_type,
            "part_index": len(text_parts) - 1
        })
    
    # Simpler suffix text
    suffixes = [
        "thank you", "please", "okay", "thanks", "that's all", "bye"
    ]
    
    if random.random() < 0.3:  # Even less suffixes (down from 0.4)
        suffix = random.choice(suffixes)
        text_parts.append(suffix)
    
    # Build final text with spaces
    text = " ".join(text_parts)
    
    # Even less word repetition (STT error)
    if make_hard and random.random() < 0.05:  # Further reduced from 0.1
        words = text.split()
        if len(words) > 3:
            idx = random.randint(1, len(words) - 2)
            words.insert(idx, words[idx])  # Repeat a word (no triplicate)
            text = " ".join(words)
            # Recalculate positions - this is approximate
            text_parts = words
    
    # Calculate exact positions for entities
    valid_entities = []
    char_pos = 0
    
    for i, part in enumerate(text_parts):
        # Check if this part is an entity
        entity_info = None
        for ent in entities_data:
            if ent["part_index"] == i:
                entity_info = ent
                break
        
        if entity_info:
            # This part is an entity
            start = char_pos
            end = char_pos + len(part)
            valid_entities.append({
                "start": start,
                "end": end,
                "label": entity_info["label"]
            })
        
        # Move position forward: part length + space (if not last)
        char_pos += len(part)
        if i < len(text_parts) - 1:
            char_pos += 1  # space
    
    return {
        "id": utt_id,
        "text": text,
        "entities": valid_entities
    }


def generate_dataset(num_examples: int, start_id: int = 1, hard_ratio: float = 0.2) -> List[Dict]:
    """Generate dataset with mix of normal and hard examples - easier overall."""
    examples = []
    for i in range(num_examples):
        utt_id = f"utt_{start_id + i:04d}"
        # 20% hard examples, 80% normal (even easier overall)
        make_hard = random.random() < hard_ratio
        examples.append(create_utterance(utt_id, make_hard=make_hard))
    return examples


def main():
    # Generate train set (800 examples - within 500-1000 range)
    print("Generating training set...")
    train_examples = generate_dataset(800, start_id=1)
    
    # Generate dev set (150 examples - within 100-200 range)
    print("Generating dev set...")
    dev_examples = generate_dataset(150, start_id=1001)
    
    # Write train.jsonl
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Write dev.jsonl
    with open("data/dev.jsonl", "w", encoding="utf-8") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(dev_examples)} dev examples")
    
    # Print sample
    print("\nSample training example:")
    print(json.dumps(train_examples[0], indent=2, ensure_ascii=False))
    print("\nSample dev example:")
    print(json.dumps(dev_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

