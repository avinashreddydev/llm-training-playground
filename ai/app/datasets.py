"""Preset training corpora for the MVP (no uploads yet).

A run may also pass ``config.dataset_text`` to train on custom text; the preset
keyed by ``run.dataset`` is the fallback / default.
"""

DATASETS: dict[str, str] = {
    "PoemBot": """Roses are red and skies are blue.
The moon shines softly over you.
A little bird sings in the tree.
The wind is dancing wild and free.
Morning light begins to glow.
Tiny flowers start to grow.
Clouds are floating, soft and slow.
Kindness is the seed we sow.
Stars are bright in velvet night.
Dreams can fly like paper kites.
Rain can tap a gentle beat.
Puddles sparkle on the street.
The sun comes up, the shadows run.
A day begins with hope and fun.
A quiet river hums a song.
It carries little leaves along.
""",
    "StoryBot": """Once upon a time, a small turtle found a golden key. It opened a tiny door under an old tree. Inside, the turtle discovered a library for animals.
One day, Maya built a robot from cardboard and tape. The robot could only say kind things. Soon, everyone wanted to build one too.
A brave squirrel wanted to touch a cloud. It climbed the tallest pine tree in the park. From the top, the cloud looked like a giant pillow.
Leo lost his red balloon at the fair. A bird carried it across the sky. The next morning, Leo found it tied to his mailbox.
A dragon lived behind the school garden. It was not scary at all. Every Friday, it helped water the tomatoes.
""",
    "ReviewBot": """Movie: The Lion King
Review: This movie is emotional, exciting, and full of memorable songs.
Movie: Frozen
Review: This movie is magical and funny, with strong characters and great music.
Movie: Toy Story
Review: This movie is creative, warm, and teaches a lesson about friendship.
Movie: Finding Nemo
Review: This movie is colorful, adventurous, and perfect for families.
Movie: Spider-Man
Review: This movie is action packed, funny, and inspiring.
Movie: Inside Out
Review: This movie is smart, creative, and helps explain feelings.
Movie: Moana
Review: This movie is beautiful, brave, and filled with adventure.
Movie: Coco
Review: This movie is touching, musical, and full of family love.
""",
    "PopcornBot": """Movie: The Lion King
Review: A heartfelt adventure with unforgettable songs and big emotions.
Score: 9/10

Movie: Frozen
Review: Magical, funny, and full of heart, with music you will sing for days.
Score: 8/10

Movie: Toy Story
Review: A warm and clever story about friendship that never gets old.
Score: 10/10

Movie: Finding Nemo
Review: A colorful ocean journey that is exciting and perfect for families.
Score: 9/10

Movie: Spider-Man
Review: Action packed and inspiring, with humor and a big beating heart.
Score: 8/10

Movie: Inside Out
Review: A smart and creative look at feelings that makes you think and smile.
Score: 9/10

Movie: Moana
Review: A brave and beautiful voyage with stunning scenes and great songs.
Score: 9/10

Movie: Coco
Review: A touching musical full of family love and gorgeous colors.
Score: 10/10
""",
    "CameronBot": """Scene 1: A young explorer finds an old map hidden inside a dusty book.
Scene 2: She follows the map into a deep forest and discovers a secret cave.

Scene 1: A robot wakes up alone in an empty factory.
Scene 2: It opens the rusty doors and steps into a bright new world outside.

Scene 1: A small boat drifts on a calm and quiet sea at sunrise.
Scene 2: A giant friendly whale rises beside it and guides the boat home.

Scene 1: A girl plants a single seed in her tiny garden.
Scene 2: By morning a tall tree has grown with glowing golden fruit.

Scene 1: A spaceship lands softly on a strange purple planet.
Scene 2: Curious aliens gather around and offer the crew a warm welcome.

Scene 1: A knight stands before a locked castle gate at midnight.
Scene 2: The gate creaks open and reveals a hall full of sleeping dragons.
""",
}

DEFAULT_DATASET = "PoemBot"


def get_dataset_text(dataset_key: str, custom_text: str | None = None) -> str:
    """Resolve the corpus for a run: custom text wins, else the preset, else default."""
    if custom_text and custom_text.strip():
        return custom_text
    return DATASETS.get(dataset_key, DATASETS[DEFAULT_DATASET])
