import numpy as np
import pandas as pd
from pysmps import smps_loader as smps

import time
from minizinc import Instance, Model, Solver
class SSLP():

    def __init__(self, data_file, scenarios, server, client, scenario):
        '''
        :param data_file: path of .cor file contain data
        :param scenario: path of .sto file contain scenarios
        '''
        self.data_file = data_file
        self.scenarios = scenarios
        self.num_server = server
        self.num_client = client
        self.num_scenario = scenario

    def read_data(self, return_str=False):
        '''
        :return: data of minizinc model or str
        '''
        data_dict = {}
        num_server = self.num_server
        num_client = self.num_client
        raw_data = smps.load_mps(self.data_file)
        obj_data = raw_data[6]
        cost = obj_data[:num_server].astype(np.int)
        revenue = (-obj_data[num_server: num_client*num_server+num_server].astype(np.int))
        if (return_str):

            str = \
            f'''
            Num_location = {num_server};
            ub_server = {num_server};
            lb_server = 0;
            Num_client = {num_client};
            cost = {cost.tolist()};
            revenue = array2d(1..{num_client}, 1..{num_server},{(revenue).tolist()});
            demand = array2d(1..{num_client}, 1..{num_server},{(revenue).tolist()});          
            capacity = {([188]*num_server   )};
            '''
            return str

        data_dict['Num_location'] = num_server
        data_dict['Num_client'] = num_client
        data_dict['cost'] = cost
        data_dict['revenue'] = revenue
        data_dict['capacity'] = [188]*num_server

        return data_dict

    def read_scenario(self):
        '''
        :return: a dictionary with {index: scenario_list}
        '''
        num_client = self.num_client
        file = open(self.scenarios)
        txt = file.readlines()[2:]
        data_dict = {}
        sc = 1  # scenario number
        idx = 0
        while (not txt[idx] == 'ENDATA\n'):
            scenario = []
            idx += 1
            for t in range(idx, idx + num_client):
                scenario.append(int(txt[t][-2]))

            data_dict[sc] = scenario
            sc += 1  # seq of scenario
            idx += num_client
        print(f'There are {sc-1} scenarios!')
        file.close()
        # dict store all scenarios
        return data_dict

    def write2dzn(self, data_dict, write_file = False):
        for name, d in data_dict.items():
            if write_file:
                with open(f'sslp/scenario_{name}.dzn', 'w') as f:
                    f.write(f'client_array = {str(d)};')  # write each scenario in corresponding .dzn file
                    f.close()


    def minizinc_model(self, sslp_model, single_scenario, info_str, add=None, solver="coin-bc"):
        '''
        :param sslp_model: path of SSLP.mzn
        :param single_scenario: one scenario list [1,0,...]
        :param solver: default solver coin-bc
        :return: dictionary of solution
        '''
        sslp = Model(sslp_model)
        sslp.add_string(info_str)
        if add:
            sslp.add_string(f'''
            server = {add};
            ''')
        coinbc = Solver.lookup(solver)
        instance = Instance(coinbc,sslp)
        instance["client_array"]=single_scenario
        result = instance.solve()
        return result

    def solve_all_scenario(self,all_scenario,info_str,write=False):
        '''
        :param all_scenario: dictionary of scenario
        :return: dictionary of scenario parameters with server solution and objective solution
        '''
        outcome = {}

        for i,s in all_scenario.items():
            result = self.minizinc_model("./SSLP.mzn",s,info_str=info_str)
            sc_set = (s,result["server"],result["objective"])
            outcome[i]=sc_set

        return outcome

    def DE(self,all_scenario,DE_model,info_data, solver="coin-bc"):
        #combine all scenarios
        client_array = [];
        for i,s in all_scenario.items():
            client_array.extend(s)
        sslp = Model(DE_model)
        sslp.add_string(info_data)
        sslp.add_string(f'''
                    Num_scenarios={self.num_scenario};
                    client_array =array2d(1..{self.num_scenario}, 1..{self.num_client}, {client_array});
                    ''')
        coinbc = Solver.lookup(solver)
        instance = Instance(coinbc, sslp)
        result = instance.solve()
        return result["server"], result["objective"]

    def EC(self,scenarios):
        # initial UB and LB and sol
        ub = float('inf')
        lb = float('-inf')
        sol = None
        while lb < ub:
            lb = 0
            S = set()  # candidate set
            # find first stage candidates (lower bound)
            for i,s in scenarios.items():
                solution = self.minizinc_model("./SSLP.mzn",s)
                x, obj = solution["server"], solution["objective"]# solve one scenario,x is 1 stage variables
                # obj is obj value
                lb += obj
                S.add(str(x))  # x_1 is first stage variable
            # evaluate 1-stage candidaties(upper bound)
            for x1 in S:
                x1 = eval(x1)
                t_ub = 0
                for j,se in scenarios.items():
                    solution_e = self.minizinc_model("./SSLP.mzn",se,add=x1)
                    obj_c = solution_e["objective"]# solve(s, scenario) one scenario from s with candidate 1-stage variable
                    t_ub += obj
                    # nogood(pdk,s)
                if t_ub < ub:
                    sol = s
                    ub = t_ub

        return sol  # 1stage variable


if __name__ == '__main__':
    # create SSLP object
    sslp = SSLP(data_file="./sslp_10_50_2000.cor", scenarios="./sslp_10_50_2000.sto",server=10,client=50,scenario=2000)
    # create scenario dictionary
    dict_sc = sslp.read_scenario() #set number of client
    info = sslp.read_data(True)
    print(info)

    ## start generate all scenario solution
    start = time.perf_counter()


    ls = sslp.solve_all_scenario(dict_sc,info);print(ls)
    # result = sslp.DE(dict_sc,"./SSLP_DE.mzn",info_data=info); print(result)

    end = time.perf_counter()
    duration = abs(end - start)
    print(f'Solved by {duration} s')