

import random


class MemoryBank:
    def __init__(self):
        self.memory = {}

    def read(self, key):
        """Retrieve a value using a specific key."""
        return self.memory.get(key, None)

    def write(self, key, value):
        """Write a value associated with a specific key."""
        self.memory[key] = value

    def delete(self, key):
        """Delete a specific key and its associated value."""
        if key in self.memory:
            del self.memory[key]

    def keys(self):
        """Get all keys stored in the memory bank."""
        return list(self.memory.keys())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.memory})'
        # return f'MemoryBank(len={len(self.memory)})'


class DNAMemoryBank(MemoryBank):
    properties = [
        # Openness to Experience
        # Inclination to engage in vivid imagination and daydreaming.
        "Imagination",
        # Appreciation for art, beauty, and aesthetic experiences.
        "Artistic Interests",
        # Openness and expression of emotions and emotional experiences.
        "Emotionality",
        # Willingness to try new experiences and engage in adventurous activities.
        "Adventurousness",
        # Openness to new ideas, intellectual curiosity, and interest in abstract thinking.
        "Intellect",
        # Importance placed on ethical, philosophical, and moral principles.
        "Liberalism",

        # Conscientiousness
        # Belief in one's own capabilities to successfully accomplish tasks and goals.
        "Self-Efficacy",
        # Preference for structure, organisation, and tidiness in one's environment.
        "Orderliness",
        # Sense of responsibility, duty, and obligation towards fulfilling tasks and commitments.
        "Dutifulness",
        # Drive for success, setting high personal standards, and working diligently towards goals.
        "Achievement Striving",
        # Ability to control impulses, maintain focus, and persevere in the face of challenges.
        "Self-Discipline",
        # Inclination to think carefully, consider alternatives, and approach decisions cautiously.
        "Cautiousness",

        # Extraversion
        # Inclination to be warm, affable, and friendly towards others.
        "Friendliness",
        # Enjoyment of being in social situations and seeking out the company of others.
        "Gregariousness",
        # Tendency to express opinions, desires, and needs confidently and directly.
        "Assertiveness",
        # Preference for being active, energetic, and engaged in physical and mental pursuits.
        "Activity Level",
        # Inclination to seek out novel, thrilling, and stimulating experiences.
        "Excitement Seeking",
        # Tendency to experience positive emotions, joyfulness, and optimism.
        "Cheerfulness",

        # Agreeableness
        # Tendency to believe in the sincerity and trustworthiness of others.
        "Trust",
        # Importance placed on adhering to ethical principles and values.
        "Morality",
        # Extent to which a person is inclined to selflessly help and support others.
        "Altruism",
        # Willingness to work harmoniously with others and avoid conflicts.
        "Cooperation",
        # Inclination to downplay one's own achievements and avoid self-promotion.
        "Modesty",
        # Ability to understand and feel compassion for others' emotions and experiences.
        "Sympathy",

        # Neuroticism
        "Anxiety",  # Tendency to experience worry, unease, and nervousness.
        # Proneness to experience feelings of anger, irritation, and hostility.
        "Anger",
        # Tendency to experience sadness, low mood, and feelings of hopelessness.
        "Depression",
        # Level of self-awareness and concern about how one is perceived by others.
        "Self-Consciousness",
        # Inclination to engage in excessive or impulsive behaviour.
        "Immoderation",
        # Sensitivity to stress, emotional reactivity, and susceptibility to negative emotions.
        "Vulnerability",

        # VIA Classification of strengths
        "Creativity",  # Ability to come up with new and original ideas.
        "Curiosity",  # Desire to learn and know more.
        "Judgment",  # Ability to think clearly and rationally.
        "Love of Learning",  # Joy of learning and acquiring knowledge.
        "Perspective",  # Ability to look at things in a broader way.
        "Bravery",  # Courage to face challenges.
        "Perseverance",  # Persistence in doing something despite difficulty.
        "Honesty",  # Being truthful and sincere.
        "Zest",  # Approach to life with excitement and energy.
        "Love",  # Affection towards others.
        "Kindness",  # Being considerate and generous.
        # Ability to understand and manage social situations.
        "Social Intelligence",
        "Teamwork",  # Work collaboratively with others.
        "Fairness",  # Treating everyone equally.
        "Leadership",  # Ability to guide and inspire others.
        "Forgiveness",  # Ability to let go of grudges and bitterness.
        "Humility",  # Modest view about one's own importance.
        "Prudence",  # Being wise in practical affairs.
        "Self-Regulation",  # Control one's behavior, emotions, and thoughts.
        # Recognizing and appreciating beauty, excellence, and skilled performance in various domains of life.
        "Appreciation of Beauty and Excellence",
        "Gratitude",  # Feeling thankful for the good things in life.
        "Hope",  # Optimistic attitude towards the future.
        # Ability to perceive, enjoy, or express what is amusing or comical.
        "Humor",
        "Spirituality",  # Deep values and meanings by which people live.

        # Dark Triad
        "Machiavellianism",  # Manipulative and exploitative tendencies.
        "Narcissism",  # Arrogance and a need for admiration.
        "Psychopathy"  # Impulsive and callous behavior.
    ]

    def __init__(self, property_values=None, default_size=8):
        super().__init__()
        if property_values is None:
            # if default_size, randomly sample from properties
            sampled_properties = random.sample(self.properties, default_size)
            for prop in sampled_properties:
                self.write(prop, random.randint(0, 100))
        else:
            for key, value in property_values.items():
                self.write(key, value)


class ProgramMemoryBank(MemoryBank):
    def __init__(self):
        super().__init__()

    def store_program(self, id, program):
        """Store a new program in the memory bank."""
        self.write(id, program)
