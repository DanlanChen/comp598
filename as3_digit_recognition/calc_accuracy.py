def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    # print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
    return corr,acc