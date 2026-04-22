import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN
from torch_geometric.utils import dense_to_sparse

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                tau: float = 1e-2,
                zeta: float = 1e-1,
                train_limit: int = 40,
                sample_times: int = 10,
                eval_edge_threshold: float = 0.5,
                llm_temperature: Optional[float] = None,
                verbose: bool = True,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        anchor_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float32).view(-1)
        if optimized_spatial:
            fixed_spatial_masks = [[1 if i != j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float32).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.topology_temperature = tau
        self.sparsity_weight = zeta
        self.train_limit = train_limit
        self.sample_times = sample_times
        self.eval_edge_threshold = eval_edge_threshold
        self.deterministic_edges = False
        self.llm_temperature = llm_temperature
        self.verbose = verbose
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.apply_runtime_options()
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.features = self.construct_features()
        self.topology_latent_dim = 16
        self.topology_rank = min(4, len(agent_names), self.topology_latent_dim)
        self.gnn_mu = GCN(self.features.size(1), 16, self.topology_latent_dim)
        self.gnn_sigma = GCN(self.features.size(1), 16, self.topology_latent_dim)
        self.ffn_d = torch.nn.Sequential(
            torch.nn.Linear(self.topology_latent_dim * 3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self.low_rank_weight = torch.nn.Parameter(torch.eye(self.topology_rank))
        self.topology_regularization_loss = torch.tensor(0.0)
        self.sketch_loss = torch.tensor(0.0)
        self.anchor_loss = torch.tensor(0.0)
        self.sparsity_loss = torch.tensor(0.0)

        self.anchor_spatial_masks = torch.nn.Parameter(anchor_spatial_masks, requires_grad=False)
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            role_description = self.prompt_set.get_description(node.role)
            profile = (
                f"Base: {node.llm_name or self.llm_name}\n"
                f"Role: {node.role}\n"
                f"Role description: {role_description}\n"
                "Plugin: none"
            )
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def apply_runtime_options(self):
        for node in list(self.nodes.values()) + [self.decision_node]:
            node.llm_temperature = self.llm_temperature
            node.verbose = self.verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def share_parameters_from(self, source: "Graph"):
        self.gnn_mu = source.gnn_mu
        self.gnn_sigma = source.gnn_sigma
        self.ffn_d = source.ffn_d
        self.low_rank_weight = source.low_rank_weight
        self.temporal_logits = source.temporal_logits

    def topology_parameters(self):
        params = []
        if self.optimized_spatial:
            params.extend(self.gnn_mu.parameters())
            params.extend(self.gnn_sigma.parameters())
            params.extend(self.ffn_d.parameters())
            params.append(self.low_rank_weight)
        if self.optimized_temporal:
            params.append(self.temporal_logits)
        return params

    def set_topology_train(self, mode: bool = True):
        self.gnn_mu.train(mode)
        self.gnn_sigma.train(mode)
        self.ffn_d.train(mode)

    def set_edge_sampling(self, deterministic: bool):
        self.deterministic_edges = deterministic

    def _concrete_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        if self.deterministic_edges:
            return torch.sigmoid(logits)
        eps = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        logistic_noise = torch.log(eps) - torch.log1p(-eps)
        return torch.sigmoid((logistic_noise + logits) / self.topology_temperature)

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def _anchor_edge_index_with_task(self):
        num_nodes = self.num_nodes
        anchor_adj = self.anchor_spatial_masks.view(num_nodes, num_nodes).float()
        extended_adj = torch.zeros((num_nodes + 1, num_nodes + 1), dtype=anchor_adj.dtype, device=anchor_adj.device)
        extended_adj[:num_nodes, :num_nodes] = anchor_adj
        extended_adj[:num_nodes, num_nodes] = 1.0
        extended_adj[num_nodes, :num_nodes] = 1.0
        edge_index, _ = dense_to_sparse(extended_adj)
        return edge_index

    def construct_learned_spatial_logits(self, query: str) -> torch.Tensor:
        device = next(self.gnn_mu.parameters()).device
        features = self.features.to(device=device, dtype=torch.float32)
        task_embedding = torch.tensor(get_sentence_embedding(query), dtype=torch.float32, device=device).unsqueeze(0)
        extended_features = torch.cat([features, task_embedding], dim=0)
        edge_index = self._anchor_edge_index_with_task().to(device)

        mu = self.gnn_mu(extended_features, edge_index)
        log_sigma = torch.clamp(self.gnn_sigma(extended_features, edge_index), min=-5.0, max=2.0)
        if self.deterministic_edges:
            latent = mu
        else:
            latent = mu + torch.randn_like(mu) * torch.exp(log_sigma)

        agent_latent = latent[:self.num_nodes]
        task_latent = latent[self.num_nodes].unsqueeze(0).repeat(self.num_nodes * self.num_nodes, 1)
        source_latent = agent_latent.repeat_interleave(self.num_nodes, dim=0)
        target_latent = agent_latent.repeat(self.num_nodes, 1)
        edge_inputs = torch.cat([source_latent, target_latent, task_latent], dim=1)
        sketch_logits = self.ffn_d(edge_inputs).view(self.num_nodes, self.num_nodes)
        sketch_probs = torch.nan_to_num(self._concrete_sigmoid(sketch_logits), nan=0.5, posinf=1.0, neginf=0.0)

        try:
            z = torch.linalg.svd(sketch_probs.detach(), full_matrices=False).U[:, :self.topology_rank]
        except RuntimeError:
            z = torch.eye(self.num_nodes, self.topology_rank, dtype=sketch_probs.dtype, device=sketch_probs.device)
        z = z.detach()
        refined_scores = z @ self.low_rank_weight @ z.t()

        anchor_adj = self.anchor_spatial_masks.view(self.num_nodes, self.num_nodes).to(device=device, dtype=refined_scores.dtype)
        candidate_adj = self.spatial_masks.view(self.num_nodes, self.num_nodes).to(device=device, dtype=refined_scores.dtype)
        refined_scores = refined_scores * candidate_adj
        sketch_probs = sketch_probs * candidate_adj
        self.sketch_loss = 0.5 * torch.linalg.matrix_norm(refined_scores - sketch_probs, ord="fro").pow(2)
        self.anchor_loss = 0.5 * torch.linalg.matrix_norm(refined_scores - anchor_adj, ord="fro").pow(2)
        self.sparsity_loss = torch.linalg.matrix_norm(self.low_rank_weight, ord="nuc")
        self.topology_regularization_loss = self.sketch_loss + self.anchor_loss + self.sparsity_weight * self.sparsity_loss
        return torch.flatten(refined_scores)

    def prepare_spatial_logits(self, query: str):
        if self.optimized_spatial:
            self.spatial_logits = self.construct_learned_spatial_logits(query)
            return

        self.spatial_logits = torch.zeros(len(self.potential_spatial_edges), dtype=torch.float32)
        self.topology_regularization_loss = torch.tensor(0.0)
        self.sketch_loss = torch.tensor(0.0)
        self.anchor_loss = torch.tensor(0.0)
        self.sparsity_loss = torch.tensor(0.0)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature).clamp(1e-6, 1 - 1e-6)
                if threshold is not None:
                    keep_edge = bool(edge_prob >= threshold)
                else:
                    keep_edge = bool(torch.rand(1, device=edge_prob.device) < edge_prob)
                if keep_edge:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature).clamp(1e-6, 1 - 1e-6)
            if threshold is not None:
                keep_edge = bool(edge_prob >= threshold)
            else:
                keep_edge = bool(torch.rand(1, device=edge_prob.device) < edge_prob)
            if keep_edge:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))

    def _debug_node_label(self, node: Node) -> str:
        return f"{node.id}({node.role})"

    def _debug_topology_snapshot(self, round: int):
        nodes = [
            {
                "node_id": node_id,
                "role": node.role,
                "node_type": node.node_name,
            }
            for node_id, node in self.nodes.items()
        ]

        spatial_edges = []
        for node in self.nodes.values():
            for successor in node.spatial_successors:
                if successor.id in self.nodes:
                    spatial_edges.append({
                        "source_id": node.id,
                        "source_role": node.role,
                        "target_id": successor.id,
                        "target_role": successor.role,
                    })

        temporal_edges = []
        for node in self.nodes.values():
            for successor in node.temporal_successors:
                if successor.id in self.nodes:
                    temporal_edges.append({
                        "source_id": node.id,
                        "source_role": node.role,
                        "target_id": successor.id,
                        "target_role": successor.role,
                    })

        return {
            "round": round + 1,
            "nodes": nodes,
            "spatial_edges": spatial_edges,
            "temporal_edges": temporal_edges,
        }

    def _debug_graph_config(self):
        num_nodes = self.num_nodes
        return {
            "optimized_spatial": self.optimized_spatial,
            "optimized_temporal": self.optimized_temporal,
            "topology_temperature": self.topology_temperature,
            "sparsity_weight": self.sparsity_weight,
            "train_limit": self.train_limit,
            "sample_times": self.sample_times,
            "eval_edge_threshold": self.eval_edge_threshold,
            "deterministic_edges": self.deterministic_edges,
            "llm_temperature": self.llm_temperature,
            "anchor_spatial_adj": self.anchor_spatial_masks.view(num_nodes, num_nodes).int().tolist(),
            "candidate_spatial_edges": int(self.spatial_masks.sum().item()),
            "candidate_temporal_edges": int(self.temporal_masks.sum().item()),
            "sketch_loss": float(self.sketch_loss.detach().cpu()),
            "anchor_loss": float(self.anchor_loss.detach().cpu()),
            "sparsity_loss": float(self.sparsity_loss.detach().cpu()),
        }

    def _debug_print_topology(self, round: int, num_rounds: int):
        topology = self._debug_topology_snapshot(round)
        if not self.verbose:
            return topology
        print("# Topology Nodes:")
        for node in topology["nodes"]:
            print(f"#   {node['node_id']}: role={node['role']}, type={node['node_type']}")

        print("# Spatial Edges:")
        if topology["spatial_edges"]:
            for edge in topology["spatial_edges"]:
                print(f"#   {edge['source_id']}({edge['source_role']}) -> {edge['target_id']}({edge['target_role']})")
        else:
            print("#   <none>")

        should_print_temporal = num_rounds > 1 or bool(topology["temporal_edges"])
        if should_print_temporal:
            print(f"# Temporal Edges:")
            if topology["temporal_edges"]:
                for edge in topology["temporal_edges"]:
                    print(f"#   {edge['source_id']}({edge['source_role']}) -> {edge['target_id']}({edge['target_role']})")
            elif round == 0:
                print("#   <none in first round>")
            else:
                print("#   <none>")
        return topology


    def run(self, inputs: Any, 
                  num_rounds:int = 1,
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.prepare_spatial_logits(inputs['task'])
        spatial_threshold = self.eval_edge_threshold if self.deterministic_edges and self.optimized_spatial else None
        log_probs += self.construct_spatial_connection(threshold=spatial_threshold)
        for round in range(num_rounds):
            temporal_threshold = self.eval_edge_threshold if self.deterministic_edges and self.optimized_temporal else None
            log_probs += self.construct_temporal_connection(round, threshold=temporal_threshold)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 1,
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.execution_trace = {"rounds": [], "final_decision": None}
        for node in self.nodes.values():
            node.execution_trace = []
        self.decision_node.execution_trace = []

        self.prepare_spatial_logits(input['task'])
        self.execution_trace["graph_config"] = self._debug_graph_config()
        spatial_threshold = self.eval_edge_threshold if self.deterministic_edges and self.optimized_spatial else None
        log_probs += self.construct_spatial_connection(threshold=spatial_threshold)

        self._log(f"\n{'#'*80}")
        self._log(f"# NEW TASK STARTED")
        self._log(f"# Task: {input['task']}")
        self._log(f"# Num Agents: {len(self.nodes)}")
        self._log(f"# Rounds: {num_rounds}")
        self._log(f"# Optimized Spatial: {self.optimized_spatial}")
        self._log(f"# Spatial Logits Range: [{self.spatial_logits.min():.3f}, {self.spatial_logits.max():.3f}]")
        self._log(f"{'#'*80}\n")

        for round in range(num_rounds):
            self._log(f"\n{'='*80}")
            self._log(f"# ROUND {round + 1}/{num_rounds}")
            self._log(f"{'='*80}")
            temporal_threshold = self.eval_edge_threshold if self.deterministic_edges and self.optimized_temporal else None
            log_probs += self.construct_temporal_connection(round, threshold=temporal_threshold)
            topology = self._debug_print_topology(round, num_rounds)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            round_trace = {
                "round": round + 1,
                "topology": topology,
                "agent_executions": [],
                "executed_order": [],
            }
            executed_order = []
            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                current_node = self.nodes[current_node_id]
                executed_order.append(self._debug_node_label(current_node))
                self._log(f"\n>> [Executing Node] ID: {current_node_id}, Role: {current_node.role}, Type: {current_node.node_name}")
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(current_node.async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        if current_node.execution_trace:
                            round_trace["agent_executions"].append(current_node.execution_trace[-1])
                        break
                    except Exception as e:
                        self._log(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            self._log(f"# Executed Order: {' -> '.join(executed_order) if executed_order else '<none>'}")
            round_trace["executed_order"] = executed_order
            self.execution_trace["rounds"].append(round_trace)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        if self.decision_node.execution_trace:
            self.execution_trace["final_decision"] = self.decision_node.execution_trace[-1]
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
