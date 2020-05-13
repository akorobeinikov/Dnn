import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class RandomNormal(Op):
    op = 'RandomNormal'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 0,
            'out_ports_count': 1,

        }
        super().__init__(graph, mandatory_props, attrs)
    '''
    def supported_attrs(self):
        return [
            'shape',
        ]
    '''
    @staticmethod
    def infer(node: Node):
        lst = str(node.get_attrs()["pb"]).split("\n")
        shape = []
        for i in lst:
            if (len(i.split("ints: ")) == 2):
                shape.append(int(i.split("ints: ")[1]))
        print(shape)

        node.out_node().shape = np.ndarray(shape)
        node.out_node().value = np.random.randn(*shape)


