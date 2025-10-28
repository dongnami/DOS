

def get_word_index(prompt, word, tokenizer):
    input_ids = tokenizer(prompt).input_ids
    word_ids = tokenizer(word, add_special_tokens=False).input_ids

    indices = []
    decoded_words = []
    for i, id in enumerate(input_ids):
        if id in word_ids:
            indices.append(i)
            decoded_words.append(tokenizer.decode(id))
    indices = find_matching_indices(indices, decoded_words, word)
    return indices


def get_object_indices(prompt, objects, tokenizer):
    object_indices = []
    for object in objects:
        indices = get_word_index(prompt.lower(), object.lower(), tokenizer)
        object_indices.append(indices)
        print(f"Word indices for '{object}': {indices}")
    return object_indices


def find_matching_indices(indices, decoded_words, target_word):
    target_word = target_word.replace(" ", "") # we don't include spacebar in the target word.
    if not indices or not decoded_words:
        return []
    
    groups = []
    current_group_indices = [indices[0]]
    current_group_words = [decoded_words[0]]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group_indices.append(indices[i])
            current_group_words.append(decoded_words[i])
        else:
            groups.append((current_group_indices, current_group_words))
            current_group_indices = [indices[i]]
            current_group_words = [decoded_words[i]]
    groups.append((current_group_indices, current_group_words))

    for group_indices, group_words in groups:
        n = len(group_words)
        for i in range(n):
            concatenation = ""
            for j in range(i, n):
                concatenation += group_words[j]
                if concatenation == target_word:
                    return group_indices[i:j+1]
    return []
