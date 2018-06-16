import random


def validate_scaling(original_data_dict, scaled_data_dict, n_samples=100):
    """Validates the result of scaling.

    Ensures by random sampling that for any number of random samples, the
    relation between both should be the same in the scaled and unscaled sets.
    """
    print "Validating scaled dataset ..."
    for i in range(n_samples):
        person1 = random.choice(original_data_dict.keys())
        person2 = random.choice(original_data_dict.keys())
        feature = random.choice(original_data_dict[person1].keys())
        person1_original_value = original_data_dict[person1][feature]
        person2_original_value = original_data_dict[person2][feature]
        person1_scaled_value = scaled_data_dict[person1][feature]
        person2_scaled_value = scaled_data_dict[person2][feature]
        try:
            assert (person1_original_value == "NaN"
                    or person2_original_value == "NaN" or cmp(
                        person1_original_value, person2_original_value) == cmp(
                            person1_scaled_value, person2_scaled_value))
        except AssertionError:
            print "{} vs. {} == {} vs. {}".format(
                person1_original_value, person2_original_value,
                person1_scaled_value, person2_scaled_value)
            raise
    print "... passed."
    return True
