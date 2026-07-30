"""Language demo: reading, fact extraction, association, generation.

Run:  python examples/demo_language.py
"""

from neurosense import Brain

CORPUS = """
The sun is a star. Stars can shine. The sun has heat and light.
A planet is a world. Earth is a planet. Earth has oceans and mountains.
Oceans have water. Water can flow. A river is a stream of water.
Rivers can flow into oceans. The moon is a satellite.
Satellites can orbit planets. The moon can orbit Earth.
Gravity is a force. Forces can pull objects. Gravity can pull the moon.
"""


def main():
    brain = Brain(name="reader")

    facts = brain.read(CORPUS)
    print(f"Learned {len(facts)} facts from reading:")
    for fact in facts[:8]:
        print("  ", fact)

    print("\nReasoning (with inheritance):")
    print("  What does the sun do? ->", brain.reason("sun", "can"))
    print("  What does Earth have? ->", brain.reason("earth", "has"))

    ok, conf, why = brain.ask("moon", "can", "orbit planets")
    print(f"\nCan the moon orbit planets? -> {ok} ({conf:.0%})")
    print("  ", why)

    print("\nFree association on 'water':", brain.free_associate("water"))

    print("\nMost similar sentence to 'what pulls the moon':")
    for text, score in brain.language.most_similar("what pulls the moon", top=2):
        print(f"   ({score:.2f}) {text}")

    print("\nOriginal generated sentences:")
    for seed in ["the sun", "rivers", "gravity"]:
        print("  >", brain.language.generate(seed=seed))


if __name__ == "__main__":
    main()
