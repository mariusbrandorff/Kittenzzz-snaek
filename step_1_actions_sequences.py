"""Task: return the last history observations and the NEXT future actions.

features: (game_turns, feature_count), encoded by supplied code.
actions: (game_turns,), where actions[t] causes state t -> state t+1.
t: index of the final OBSERVED state. Both slices must include index t.
Return arrays of shapes (history, feature_count) and (future,).
The caller handles game boundaries and excludes terminal-horizon examples.
"""


def student_make_example(features, actions, t, history, future):
    # TODO: select observations ending at t and actions starting at t.
    raise NotImplementedError('Complete student_make_example in Step 1 before preparing the dataset.')


if __name__ == '__main__':
    from supplied_pipeline import prepare
    prepare(student_make_example)
