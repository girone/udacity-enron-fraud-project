from abc import ABCMeta, abstractmethod, abstractproperty
import re
import os
import numpy as np
from math import isnan
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from parse_email_text import parse_out_text


def get_without_NaN(data, person, key):
    value = float(data[person][key]) if data[person].has_key(key) else 0.
    return 0.0 if isnan(value) else value


def list_poi_names(data):
    poi_names = set([person for person, pdata in data.iteritems() if pdata["poi"] == True])
    return poi_names


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


class EmailWordFeatures(NewFeature):
    FEATURE_1 = "email_tfidf_poi_top100_distance"

    def extend(self, data_dict):
        raise Exception("Implementation not finished. See `EmailWordFeatures.extract_top_poi_words()`")
        maildir_to_person = self.create_maildir_person_map(data_dict.keys())
        # TODO(Jonas): Many persons do not seem to have a directory in the maildir/. How to handle those? Feature value NaN probably...
        person_emails = self.read_emails(maildir_to_person, data_dict)
        tfidf_average_per_person = self.vectorize_emails(person_emails)
        poi_names = list_poi_names(data_dict)
        top_poi_words = self.extract_top_poi_words(tfidf_average_per_person, poi_names, n_words=100)
        for person in data_dict.keys():
            distance = self.compute_distance_to_top_poi_words(tfidf_result, top_poi_words, person)
            data_dict[person][self.FEATURE_1] = distance
        return data_dict

    def new_feature_names(self):
        pass

    def create_maildir_person_map(self, persons):
        """Returns a dictionary where the values come from `persons`.

        For every person, the dictionary has an entry ("surname-n": "FAMILYNAME SECONDNAME NAME").
        """
        pattern = re.compile(r"(\w+) (\w)")
        maildir_names = {}
        for name, person in [(person.lower(), person) for person in persons]:
            m = pattern.search(name)
            if m:
                maildir_names["{}-{}".format(m.group(1), m.group(2))] = person
        print maildir_names
        return maildir_names

    def read_and_preprocess_emails(self, mail_file_name):
        with open(mail_file_name, "r") as mail_file:
            words = parse_out_text(mail_file)
            return words
        print "No words found in", mail_file_name
        exit(1)

    def read_emails(self, maildir_to_person, data_dict):
        devCounter = 0
        words_and_documents_per_person = defaultdict(dict)
        # mail file paths have the format "../maildir/<username>/..."
        pattern = re.compile(r"\.\./maildir/(\w+-\w)(/|$)")
        without_match = set()
        with_match = set()
        not_processed = set()
        for root, _, filenames in os.walk("../maildir"):
            # if devCounter > 10000:
            #     break
            if filenames:
                # parse the person's name
                m = pattern.match(root)
                if not m:
                    print "could not process directory", root
                    not_processed.add(root)
                    continue
                maildir = m.group(1)
                # skip uninteresting persons
                if maildir not in maildir_to_person:
                    without_match.add(maildir)
                    continue
                person = maildir_to_person[maildir]
                with_match.add(maildir)
            for filename in filenames:
                mail_file_name = os.path.join(root, filename)
                words = self.read_and_preprocess_emails(mail_file_name)
                words_and_documents_per_person[person][filename] = words
                devCounter += 1
                # if devCounter > 10000:
                #     exit(1)
                #     return words_and_documents_per_person  # TODO(Jonas): remove this
        print
        print "=== processing summary ==="
        for maildir in with_match:
            print "'{}' was matched with '{}'".format(maildir, maildir_to_person[maildir])
            maildir_to_person.pop(maildir, 0)
        print
        for maildir, person in sorted(maildir_to_person.items()):
            print "'{}'/'{}' could not be matched with any maildir, email is '{}'".format(person, maildir, data_dict[person]["email_address"])
        print
        for maildir in sorted(without_match):
            print "username/maildir '{}' could not be matched with a person from the Enron dataset".format(maildir)
        print
        for maildir in sorted(not_processed):
            print "could no process {}".format(maildir)
        print
        print "Remember that some persons have been removed as outliers from the test input (e.g. Kenneth Lay)."
        return words_and_documents_per_person

    @staticmethod
    def summarize_document_scores(X):
        """Summarizes a set of document scores.

        Uses a simple average: For every word, it averages the tf.idf over the
        total number of documents. Thus, if documents where the word does not
        occur do count into the average with 0.

        In the future, this could take a second argument to control the
        aggregation method.
        """
        return sum(X) / X.shape[0]

    def vectorize_emails(self, mails_per_person):
        """Returns a dictionary {person: [(tf.idf score, word index)]}."""
        average_word_scores = {}
        for person, documents in mails_per_person.iteritems():
            print person
            vectorizer = TfidfVectorizer(stop_words="english")
            X = vectorizer.fit_transform(documents.values())
            average_word_scores[person] = self.summarize_document_scores(X)
        return average_word_scores

    def extract_top_poi_words(self, tfidf_average_per_person, poi_names, n_words, sorted=True):
        """Returns a sparse matrix of tuples (tf.idf score, word_index)."""
        from scipy.sparse import csr_matrix
        poi_scores = []
        for person in poi_names:
            try:
                poi_scores.append(tfidf_average_per_person[person])
            except KeyError:
                pass  # not all POIs have email data available
        try:
            average_over_pois = sum(poi_scores) / len(poi_scores)
        except ValueError:
            import pdb; pdb.set_trace()
            # NOTE(Jonas): This is a dead end, because there is too few data available.
            # The emails are available for just 20 out of 146 persons in the data set,
            # and just for four POIs. These are too few to compute a meaningful average
            # of the tf.idf scores and use it to measure how far other persons fall off
            # this average. It's a pity those mails are missing, I wonder what other
            # persons these emails belong to.
        print average_over_pois
        return []

    def compute_distance_to_top_poi_words(self, tf_idf_per_person, top_poi_words, person):
        """Computes the Euklidean distance (sqrt(sum((A-B)^2))).

        This operation is not associative: Words which are in the persons
        dictionary but not in the POI words are ignored.
        """
        scores_for_person = tf_idf_per_person[person]
        euklid = 0
        for word, score in top_poi_words:
            if word in scores_for_person:
                person_score = scores_for_person[word]
                diff = score - person_score
                euklid += diff * diff
            else:
                euklid += score * score
        return sqrt(euklid)


def main():
    """Development test environment."""
    # Read persons in the dataset, so that others may be skipped.
    from tester import load_classifier_and_data
    clf, dataset, feature_list = load_classifier_and_data()
    # Run the email feature computation.
    generator = EmailWordFeatures()
    generator.extend(dataset)

if __name__ == "__main__":
    main()