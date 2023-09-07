# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights
from models.mnist_cnn import mnist_lenet, mnist_lenet1


class Edge():

    def __init__(self, id, cids, shared_layers):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer1 = {}
        self.receiver_buffer2 = {}
        self.ll=[]
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.mix=0.5
        self.local_model=shared_layers.state_dict()
        self.clock = []

    def refresh_edgeserver(self):
        self.receiver_buffer1.clear()
        self.receiver_buffer2.clear()
        del self.id_registration[:]
        del self.ll[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] =len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict,v,a):
        self.receiver_buffer1[client_id] = cshared_state_dict
        self.receiver_buffer2[client_id] = v
        self.ll.append(a)
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict1 = [dict for dict in self.receiver_buffer1.values()]
        received_dict2 = [dict for dict in self.receiver_buffer2.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict1,
                                                 s_num= sample_num)
        self.local_model =average_weights(w = received_dict2,
                                                 s_num= sample_num)
        self.mix = sum(self.ll)/len(self.ll)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict),self.mix,copy.deepcopy(self.local_model))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict=copy.deepcopy(
                                    self.shared_state_dict))

        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

