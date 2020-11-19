import networkx as nx
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from matplotlib import cm
import sys
import json

colormap = cm.get_cmap('jet')

attrib_file = open("attributes.txt", encoding="utf8")
lines = attrib_file.readlines()
attrs = []

for i in range(int(len(lines)/4)):
    idx = 4*i
    attrs.append({
        'title': lines[idx].strip().split(',')[1],
        'prob_multi': float(lines[idx].strip().split(',')[0]),
        'keys': lines[idx+1].strip().split(','),
        'letters': lines[idx+2].strip().split(','),
        'choice_prob': [float(p) for p in lines[idx+3].strip().split(',')] 
    })

legend = {}
for attr in attrs:
    title = attr['title'] 
    legend[title] = []
    if title == 'Reaction':
        legend['dark'] = []

    for i, key in enumerate(attr['keys']):
        color = colormap(i/len(attr['keys']))
        if i == 0:
            color = [0.46,0.13,0.46]
        ctxt = "rgb(" + str(color[0] * 200) + "," + str(color[1] * 200) + "," + str(color[2] * 200) + ")"
        legend[title].append([key,ctxt])
        if attr['title'] == "Reaction":
            ctxt2 = "rgb(" + str(color[0] * 120) + "," + str(color[1] * 120) + "," + str(color[2] * 120) + ")"
            legend['dark'].append([key,ctxt2])

json_object = json.dumps(legend, indent = 4) 
  
# Writing to sample.json 
with open("viz/legend.json", "w") as outfile: 
    outfile.write(json_object) 

# sys.exit(0)

def random_planet():
    planet = {}
    colored_planet = {}
    for attr in attrs:
        attribs = []
        multi = attr['prob_multi']
        index = random.choices(range(len(attr['keys'])), attr['choice_prob'])[0]
        attribs.append(attr['keys'][index])
        color = colormap(index/len(attr['keys']))
        if index == 0:
             color = [0.46,0.13,0.46]
        while random.random() < multi:
            index = random.choices(range(len(attr['keys'])), attr['choice_prob'])[0]
            att = attr['keys'][index]
            if att not in attribs:
                attribs.append(att)
        planet[attr['title']] = attribs
        ctxt = "rgb(" + str(color[0] * 128) + "," + str(color[1] * 128) + "," + str(color[2] * 128) + ")"
        if attr['title'] == "Reaction":
            ctxt2 = "rgb(" + str(color[0] * 40) + "," + str(color[1] * 40) + "," + str(color[2] * 40) + ")"
            colored_planet['dark'] = ctxt2
        colored_planet[attr['title']] = ctxt
    return planet, colored_planet


fp = open("name_db/cities5000.txt", encoding="utf8")
cities = list(enumerate(fp))

def random_name():
    return random.choice(cities)[1].split('\t')[1]
    
# Le code est: nord N, R ou O, sud S, D ou U, équateur E, T ou K, aléatoire A, I ou L				
cons = [["N","R"],["S","D"],["T","K"],["L"]]
vow = [["O"],["U"],["E"],["A","I"]]

def next_id(parent_id, pos):
    t = cons[pos]
    if len(parent_id) != 0 and any(parent_id[-1] in s for s in cons):
        t = vow[pos]
    return parent_id + random.choice(t)


def init_node(node_idx):
    node = G.nodes[node_idx]
    node['name'] = random_name()
    planet, colored_planet = random_planet()
    node['planet'] = planet
    # for key in colored_planet:
    #     node[key] = colored_planet[key]


def bfs(G, node_idx, reach_proba = 0.8):
    node_idxs = [node_idx]
    index = 0
    while index < len(node_idxs):
        parent_idx = node_idxs[index]
        node = G.nodes[parent_idx]

        edges = list(G.edges(parent_idx, data=True))
        random.shuffle(edges)
        # print("**", parent_idx)
        for _,t,d in edges:
            target = G.nodes[t]
            if(node['reached'] == 1 and target['reached'] == 1 \
                and d['unreached'] == 0 and node['pred'] == -1):
                node['pred'] = t
                

        #     # print("target", t, d['unreached'])
        #     if t in node_idxs[0:index] and d['unreached'] == 0:
        #         if(node['pred'] == -1):
        #             if( target['pred'] >= 0 or t == 0 ):
        #                 node['pred'] = t
                        # print("set predec", parent_idx, "to", t)
                # elif target['pred'] < 0 and node['pred'] >= 0:
                #     target['pred'] = parent_idx
                #     print("SET predec", t, "to", parent_idx)
                # else:
                #     print("PROBLEM predec", parent_idx, "to", t)
                

        
        ## add randomly successor to list and init
        for pos, edge in enumerate(G.edges(parent_idx, data=True)):
            _, t, d = edge
            target = G.nodes[t]

            if d['unreached'] == -1:
                if node["reached"] == 0:
                    d['unreached'] = 1
                else:
                    d['unreached'] = 0 if random.random() < reach_proba else 1
                    if d['unreached'] == 1:
                        d['unreached'] = 0 if random.random() < reach_proba else 1

                    # print("Just setted:", parent_idx, t, d['unreached'])
                    if d['unreached'] == 0:
                        target['reached'] = 1
            elif d['unreached'] == 0:
                if target['reached'] == 1 and node['reached'] == 0:
                    node['reached'] = 1
                elif node['reached'] == 1 and target['reached'] == 0:
                    target['reached'] = 1
            
            if t not in node_idxs:
                node_idxs.insert(random.randint(index+1, len(node_idxs)+1), t) #parcours pseudo aléatoire
                init_node(t)

        index += 1


def w_historic_path(G, node_idx):
    if G.nodes[node_idx]['pred'] < 0:
        return []
    
    path = [node_idx]
    predecessor = G.nodes[node_idx]['pred']
    w = 0
    while  predecessor > 0:
        path.insert(0, predecessor)
        w += G.edges[(predecessor, G.nodes[predecessor]['pred'])]["unreached"]
        predecessor = G.nodes[predecessor]['pred']
    path.insert(0, 0)
    return (w, path)

def historic_path(G, node_idx):
    if G.nodes[node_idx]['pred'] < 0:
        # print(node_idx, "has no predec")
        return []
    
    path = [node_idx]
    predecessor = G.nodes[node_idx]['pred']
    while  predecessor > 0:
        path.insert(0, predecessor)
        predecessor = G.nodes[predecessor]['pred']
    path.insert(0, 0)
    return path

def path_weight(G, path):
    w = 0
    for i in range(1, len(path)):
        w +=  G.edges[(path[i], path[i-1])]['unreached']
    return w


random.seed(111)
N = 350
K = 4
reach_proba = 0.4
# Graph
G = nx.random_regular_graph(K,N, seed=300)
print("Graph done")
nx.set_node_attributes(G, '', 'name')
nx.set_node_attributes(G, -1, 'pred')
nx.set_node_attributes(G, 0, 'reached')
nx.set_edge_attributes(G, -1, 'unreached')

# Graph
G.nodes[0]['name'] = 'Terre'
G.nodes[0]['pred'] = -999
G.nodes[0]['reached'] = 1

bfs(G, 0, reach_proba)
print("Init done")

import json
import flask
from networkx.readwrite import json_graph

for n in G:
    node = G.nodes[n]
    node['neighbors'] = []
    # node['neighbors_name'] = []
    edges = G.edges(n, data=True)
    for _,t,d in edges:
        node['neighbors'].append(t)
        # node['neighbors_name'].append(G.nodes[t]['name'])
    s_path = nx.single_source_dijkstra(G, source=0, target=n)
    s_path_w = path_weight(G, s_path[1])
    s_reachable_path = nx.single_source_dijkstra(G, source=0, target=n, weight='unreached')
    if s_path_w > s_reachable_path[0] and len(s_path[1]) >= len(s_reachable_path[1]):
        s_path = s_reachable_path
    else:
        s_path = (s_path_w, s_path[1])
        
    h_path = historic_path(G, n)
    # if(node['reached'] == 1):
        # print(n, node['name'], " > ", h_path, " : ", s_reachable_path, " : ", s_path)
    # if s_path[1] != s_reachable_path[1] and s_reachable_path[0] == 0 and s_reachable_path[1] != h_path:
    #     print(n, " > ", h_path, " : ", s_reachable_path, " : ", s_path)

    node['h_path'] = h_path
    node['s_path'] = s_path[1]
    node['hs_path'] = s_reachable_path[1]
    knowns = []
    unknowns = []
    for nid in s_reachable_path[1]:
        if G.nodes[nid]['reached'] == 1:
            knowns.append(nid)
        else:
            unknowns.append(nid)
    node['hs_path_kn'] = knowns
    node['hs_path_unkn'] = unknowns



d = json_graph.node_link_data(G)
json.dump(d, open("viz/multivers.json", "w"), indent=4)
print("Successfully generated multivers data ")
nb_reached = 0
for n in G:
    nb_reached += 0 if G.nodes[n]['reached'] == 0 else 1
print("Reached:", nb_reached, "planets on", N)






# # Serve the file over http to allow for cross origin requests
# app = flask.Flask(__name__, static_folder="force")


# @app.route("/")
# def static_proxy():
#     return app.send_static_file("force.html")


# print("\nGo to http://localhost:8000 to see the example\n")
# app.run(port=8000)









################ SCRAP ####################

# ### PROBE for remarkable cases
# for n in G:
#     node = G.nodes[n]
    # s_path = nx.single_source_dijkstra(G, source=0, target=n)
    # s_path_w = path_weight(G, s_path[1])
    # s_reachable_path = nx.single_source_dijkstra(G, source=0, target=n, weight='unreached')
    # if s_path_w > s_reachable_path[0] and len(s_path[1]) >= len(s_reachable_path[1]):
    #     s_path = s_reachable_path
    # else:
    #     s_path = (s_path_w, s_path[1])
        
    # h_path = historic_path(G, n)
    # if s_path[1] != s_reachable_path[1] and s_reachable_path[0] == 0 and s_reachable_path[1] != h_path:
    #     print(n, " > ", h_path, " : ", s_reachable_path, " : ", s_path)






# pos = nx.spring_layout(G)  # positions for all nodes
# nx.draw_networkx_labels(G, pos)
# plt.show()

# shells = []
# new_shell = [0]
# dist = 1
# while len(new_shell) > 0:
#     shells.append(new_shell)
#     new_shell = nx.descendants_at_distance(G, 0, dist)
#     dist += 1


# d, t = nx.single_source_dijkstra(G, 0)
# shells = [[] for i in range(max(d.values())+1)] 
# for n in G:
#     node = G.nodes[n]
#     s_path = nx.single_source_dijkstra(G, source=0, target=n)
#     shells[s_path[0]].append(n)
# pos = nx.shell_layout(G, shells)


# pos = nx.spectral_layout(G)#, weight='unreached')
# print("Layout done")

# reached_nodes = [idx for (idx, d) in G.nodes(data=True) if d["reached"] == 1]
# unreached_nodes = [idx for (idx, d) in G.nodes(data=True) if d["reached"] == 0]
# nx.draw_networkx_nodes(G, pos, nodelist=unreached_nodes, node_size=10, node_color ="lightgreen", alpha=0.5)
# nx.draw_networkx_nodes(G, pos, nodelist=reached_nodes, node_size=25, node_color ="blue", alpha=0.8)
# nx.draw_networkx_nodes(G, pos, nodelist=[0], node_size=100, node_color="red")

# reached_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["unreached"] == 0]
# unreached_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["unreached"] == 1 and \
#                     (G.nodes[u]['reached'] == 1 or G.nodes[v]['reached'] == 1)]

# nx.draw_networkx_edges(G, pos, edgelist=reached_edges, width=1, alpha=0.5, edge_color="blue", style="dashed")
# nx.draw_networkx_edges(G, pos, edgelist=unreached_edges, width=1, alpha=0.5, edge_color="green", style="dotted")

# # nx.draw(G)
# # nx.draw_random(G)
# # nx.draw_circular(G)
# # nx.draw_spectral(G)
# plt.show()
# # nx.draw(G)
# # plt.savefig("path.png")
# # nx.draw_graphviz(G)
# # nx.write_dot(G,'file.dot')










# from mayavi import mlab
# import numpy as np
# pos = nx.spring_layout(G, dim=3)
# xyz = np.array([pos[v] for v in sorted(G)])
# scalars = np.array(list(G.nodes())) + 5
# pts = mlab.points3d(
#     xyz[:, 0],
#     xyz[:, 1],
#     xyz[:, 2],
#     scalars,
#     scale_factor=0.1,
#     scale_mode="none",
#     colormap="Blues",
#     resolution=20,
# )

# pts.mlab_source.dataset.lines = np.array(list(G.edges()))
# tube = mlab.pipeline.tube(pts, tube_radius=0.01)
# mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
# mlab.show()