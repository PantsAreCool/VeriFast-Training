# Task 2: Build a Simple BPE Tokenizer
# Implement a BPE tokenizer from scratch. Start with a training corpus of at least 100 words (you can use a paragraph from any article). 
# Train the tokenizer with 30 merge operations. 
# Then implement an encode function that applies the learned merges to new text, and a decode function that reverses the process. 
# Test your tokenizer on 3 unseen sentences and show the tokenization results. 
# How does your tokenizer handle words that were not in the training corpus?

training_corpus = (
            "As you come into this world, something else is also born. You begin your life, "
            "and it begins a journey towards you. It moves never faster, never slower, perhaps "
            "a thousand miles away, but it never stops. It never hurries, it never turns aside. "
            "From that moment on, if you look in a mirror, you will see a shadow behind you. "
            "Only the Second Click. Only the clock ticking. And when you arrive at your "
            "destination, it will be standing there waiting. You can outrun it, you can trick "
            "it, you can hide, but one day, you will be in a room and there will be no way out, "
            "no doors, no windows. And it will walk in. And you will understand. "
            "Clara, my Clara, I will always remember you. The stars are going out, the universe "
            "is turning to dust, but this room stays. The wall is twelve feet pure diamond. "
            "Harder than anything else in the cosmos. And I am going to break it down with my "
            "bare fists. If it takes me a day, a year, or ten billion years, I will get through."
        )

new_text = (
            "There's this emperor, and he asks the shepherd's boy how many seconds in eternity. "
            "And the shepherd's boy says, 'There's this mountain of pure diamond. " 
            "It takes an hour to climb it and an hour to go around it, and every hundred years a little "
            "bird comes and sharpens its beak on the diamond mountain. And when the entire mountain is chiseled away, "
            "the first second of eternity will have passed.' You may think that's a hell of a long time. "
            "Personally, I think that's a hell of a bird."
)