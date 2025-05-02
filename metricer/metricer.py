

class Metricer:
    def __init__(self, config):
        self.config = config
    
    def _compress(self, train_loader, test_loader, model_I, model_T):
        re_BI = list([])
        re_BT = list([])
        re_L = list([])
        for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
            with torch.no_grad():
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            re_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            re_BT.extend(code_T.cpu().data.numpy())
            re_L.extend(data_L.cpu().data.numpy())
        qu_BI = list([])
        qu_BT = list([])
        qu_L = list([])
        for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
            with torch.no_grad():
                var_data_I = Variable(data_I.cuda())
                _, code_I = model_I(var_data_I)
            code_I = torch.sign(code_I)
            qu_BI.extend(code_I.cpu().data.numpy())

            var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
            _, code_T = model_T(var_data_T)
            code_T = torch.sign(code_T)
            qu_BT.extend(code_T.cpu().data.numpy())
            qu_L.extend(data_L.cpu().data.numpy())
        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)

        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)
        return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

    def calculate_hamming(B1, B2):
        leng = B2.shape[1]  # max inner product value
        distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
        return distH

    def eval_mAP_all(self, query_HashCode, retrieval_HashCode, query_Label, retrieval_Label):
        num_query = qu_L.shape[0]
        map = 0
        for iter in range(num_query):
            gnd = (np.dot(query_Label[iter, :], retrieval_Label.transpose()) > 0).astype(np.float32)
            tsum =np.int(np.sum(gnd))
            if tsum == 0:
                continue
            hamm = self.calculate_hamming(query_HashCode[iter, :], retrieval_HashCode)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
            tindex = np.asarray(np.where(gnd == 1)) + 1.0
            map_ = np.mean(count / (tindex))
            map = map + map_
        map = map / num_query
        return map