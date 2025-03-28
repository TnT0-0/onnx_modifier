from collections import defaultdict
from onnx import helper, TensorProto, AttributeProto, shape_inference
import onnx


class ModelModifier:
    def __init__(self, model):
        self.model = onnx.load(model) if isinstance(model, str) else model
        self.opset = [op for op in self.model.opset_import]
        self.node_list = [node for node in self.model.graph.node]
        self.initializer_list = [ini for ini in self.model.graph.initializer]
        
        
        self.model_input = self.model.graph.input
        self.model_output = self.model.graph.output

    def add_initializer(self, initial_vals):
        if not isinstance(initial_vals, list):
            self.initializer_list.append(initial_vals)
        else:
            self.initializer_list.extend(initial_vals)
    
    def remove_initializer(self, initializer_names):
        if not isinstance(initializer_names, list):
            initializer_names = [initializer_names]
        initial_temp_list = []
        for ini in self.initializer_list:
            if ini.name not in initializer_names:
                initial_temp_list.append(ini)
        self.initializer_list = initial_temp_list

    def add_node(self, nodes, new_output=None):
        # node_inputs_map: dict
        # key: input name
        # value: nodes list
        node_inputs_map = defaultdict(list)
        for node in self.node_list:
            for input in node.input:
                node_inputs_map[input].append(node)
        if not isinstance(nodes, list):
            nodes = [nodes]

        model_input_name = [input.name for input in self.model_input]
        model_output_name = [output.name for output in self.model_output]

        input_output_nodes = []
        for node in nodes:
            for input in node.input:
                if len(node_inputs_map[input]) == 0 and input not in model_output_name:
                    continue
                # add new node to model input
                if input in model_input_name:
                    if node not in input_output_nodes:
                        input_output_nodes.append(node)
                    # insert node
                    self.node_list.insert(0, node)
                    # append new node in node_inputs_map
                    node_inputs_map[input].append(node)
                    # modify next node input
                    for next_node in node_inputs_map[input]:
                        input_idx = list(next_node.input).index(input)
                        next_node.input[input_idx] = node.output[0]
                        # update next_node message in node_inputs_map
                        node_inputs_map[input].remove(next_node)
                        node_inputs_map[node.output[0]].append(next_node)
                    continue
                # add new node to model output
                assert input in model_output_name and new_output is not None, "If you want to modify model output, please specify new_output!"
                if node not in input_output_nodes:
                    input_output_nodes.append(node)
                # append new node
                self.node_list.append(node)
                # update node_inputs_map
                node_inputs_map[input].append(node)
                org_output = next(o for o in self.model_output if o.name == input)
                self.model_output.remove(org_output)
                self.model_output.append(new_output)

        new_nodes = []
        for node in nodes:
            if node not in input_output_nodes:
                new_nodes.append(node)

        for node in new_nodes:
            for input in node.input:
                if len(node_inputs_map[input]) == 0:
                    continue
                node_pos = []
                node_to_be_remove = defaultdict(list)
                for next_node in node_inputs_map[input]:
                    input_index = list(next_node.input).index(input)
                    next_node.input[input_index] = node.output[0]

                    node_pos.append(self.node_list.index(next_node))
                    node_to_be_remove[next_node.input[input_index]].append(next_node)
                # update node_inputs_map 
                for input_idx, nodes in node_to_be_remove.items():
                    for remove_node in nodes:
                        node_inputs_map[input].remove(remove_node)
                        node_inputs_map[input_idx].append(remove_node)
                node_inputs_map[input].append(node)
            self.node_list.insert(min(node_pos), node)

    def remove_node(self, node_names):
        if isinstance(node_names, str):
            node_names = [node_names]

        # node_inputs_map: dict
        # key: input name
        # value: nodes list
        node_inputs_map = defaultdict(list)
        for node in self.node_list:
            for input in node.input:
                node_inputs_map[input].append(node)

        node_inputs_map_copy = node_inputs_map
        node_names_map = defaultdict(list)
        for node in self.node_list:
            node_names_map[node.name] = node
        
        for node_name in node_names:
            next_node_input = node_names_map[node_name].output[0]
            for next_node in node_inputs_map_copy[next_node_input]:
                next_node.input[0] = node_names_map[node_name].input[0]
            self.node_list.remove(node_names_map[node_name])

    def modify_node(self, node_name, **kwargs):
        current_node = None
        for node in self.node_list:
            if node.name == node_name:
                current_node = node
        for attr in current_node.attribute:
            if attr.name in kwargs:
                if attr.type == AttributeProto.FLOAT:
                    attr.f = kwargs.get(attr.name)
                elif attr.type == AttributeProto.INT:
                    attr.i = kwargs.get(attr.name)
                elif attr.type == AttributeProto.STRING:
                    attr.s = kwargs.get(attr.name)
                elif attr.type == AttributeProto.TENSOR:
                    attr.t = kwargs.get(attr.name)
                elif attr.type == AttributeProto.FLOATS:
                    attr.floats[:] = kwargs.get(attr.name)
                elif attr.type == AttributeProto.INTS:
                    attr.ints[:] = kwargs.get(attr.name)
                elif attr.type == AttributeProto.STRINGS:
                    attr.strings[:] = kwargs.get(attr.name)

    def export_onnx(self, graph_name='test_graph', save_path='./new_model.onnx'):
        graph = helper.make_graph(nodes = self.node_list,
                                  name=graph_name,
                                  inputs=self.model_input,
                                  outputs=self.model_output,
                                  initializer=self.initializer_list)
        
        model = helper.make_model(graph, opset_imports=self.opset)
        model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, save_path)