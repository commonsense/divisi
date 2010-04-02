from csc.divisi.cnet import *
from numpy import *
import string

def main():
    standard = ConceptNet2DTensor.load_from_db('en')
    print standard.shape

    for concept in standard.label_list(0):
        true_row = standard.dense_slice(0, concept)
        predicted_row = standard.predict_properties(concept)
        mask = true_row.array_op(not_equal, 0)
        notmask = true_row.array_op(equal, 0)
        valid_predictions = predicted_row.array_op(multiply, notmask)
        nprops = valid_predictions.shape[0]
        for index in range(nprops):
            if valid_predictions._data[index] > 0.01:
                label = valid_predictions.label(0, index)
                label_parts = label.split('/')
                if len(label_parts) != 2: continue
                left, right = label_parts
                if left[0] in string.uppercase:
                    stem1 = concept
                    rel = left
                    stem2 = right
                else:
                    stem1 = left
                    rel = right
                    stem2 = concept
                print '%s\t%s\t%s\t%5.5f' % (stem1, rel, stem2,
                valid_predictions[label])

if __name__=='__main__':
    main()
