from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from math import isnan


def get_without_NaN(data, person, key):
    value = float(data[person][key]) if data[person].has_key(key) else 0.
    return 0.0 if isnan(value) else value


class NewFeature:
    __metaclass__ = ABCMeta

    @abstractmethod
    def extend(self, data_set):
        """Extends a data set with new features.

        Returns the data_set.
        """

    @abstractproperty
    def new_feature_names(self):
        """Returns a list of the names this class adds."""


class RelativeFeature(NewFeature):
    def __init__(self, numerator_feature, denominator_feature):
        self.numerator_feature = numerator_feature
        self.denominator_feature = denominator_feature
        self.feature_name = "relative_" + numerator_feature

    def extend(self, data_dict):
        """Extends the data set with the relation of a feature towards
        another (typically summarizing feature).
        """
        for person, data in data_dict.iteritems():
            numerator = get_without_NaN(data_dict, person,
                                        self.numerator_feature)
            denominator = get_without_NaN(data_dict, person,
                                          self.denominator_feature)
            data_dict[person][self.feature_name] = (numerator / denominator
                                                    if denominator and
                                                    not np.isnan(denominator)
                                                    else 0.)
        return data_dict

    def new_feature_names(self):
        return [self.feature_name]


class HasEnronEmailAddress(NewFeature):
    FEATURE_NAME = "has_enron_email_address"

    def extend(self, data_dict):
        """Extends the `data_dict` with a numerical email feature.

        The feature takes values 1 or 0 indicating if the person has an
        Enron email address or not.
        """
        for person, data in data_dict.iteritems():
            email_address = data["email_address"]
            data_dict[person][self.FEATURE_NAME] = (1 if email_address != "NaN"
                                                    else 0)
        return data_dict

    def new_feature_names(self):
        return [self.FEATURE_NAME]


class EmailShares(NewFeature):
    FEATURE_1 = "emails_to_poi_share"
    FEATURE_2 = "emails_from_poi_share"

    def extend(self, data_dict):
        """Extends the data set `data_dict` with the share of emails from/to POIs."""

        for person, data in data_dict.iteritems():
            to_poi = get_without_NaN(data_dict, person,
                                     "from_this_person_to_poi")
            from_total = get_without_NaN(data_dict, person, "from_messages")
            data_dict[person][self.FEATURE_1] = (to_poi / from_total
                                                 if from_total
                                                 and not np.isnan(from_total)
                                                 else 0.)
            from_poi = get_without_NaN(data_dict, person,
                                       "from_poi_to_this_person")
            to_total = get_without_NaN(data_dict, person, "to_messages")
            data_dict[person][self.FEATURE_2] = (from_poi / to_total
                                                 if to_total
                                                 and not np.isnan(to_total)
                                                 else 0.)
        return data_dict

    def new_feature_names(self):
        return [self.FEATURE_1, self.FEATURE_2]


class PaymentsStockRatio(NewFeature):
    FEATURE_PAYMENT_STOCK_RATIO = "total_payments_to_stock_value_ratio"

    def extend(self, data_dict):
        """Extends the data_dict with the share of 'total_payments' to 'total_stock_value'."""
        for person, data in data_dict.iteritems():
            payments = get_without_NaN(data_dict, person, "total_payments")
            stock_value = get_without_NaN(data_dict, person,
                                          "total_stock_value")
            ratio = (payments + 1) / (stock_value + 1)
            data_dict[person][self.FEATURE_PAYMENT_STOCK_RATIO] = ratio
        return data_dict

    def new_feature_names(self):
        return [self.FEATURE_PAYMENT_STOCK_RATIO]
