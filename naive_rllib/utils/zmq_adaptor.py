import zmq

# TODO set the time out
class ZmqAdaptor(object):
    modes = {"req": zmq.REQ, 'rep': zmq.REP, 'pub': zmq.PUB, 'sub': zmq.SUB, 'push': zmq.PUSH, 'pull': zmq.PULL,
             'router': zmq.ROUTER, 'dealer': zmq.DEALER}
    con_types = {"req": 'connect', 'rep': 'bind', 'pub': 'bind', 'sub': 'connect', 'push': 'connect', 'pull': 'bind',
                 'router': 'bind', 'dealer': 'connect'}

    def __init__(self, config: dict, logger=None):
        self.logger = logger
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self._sockets = {}
        for k, v in config.items():
            self._sockets[k] = self.create_socket(v['mode'], v['ip'], v['port'])

    def create_socket(self, mode: str, ip: str, port: int):
        assert mode in self.modes
        socket = self.context.socket(self.modes[mode])
        if self.con_types.get(mode) == "bind":
            socket.bind("tcp://%s:%d" % (ip, port))
        else:
            socket.connect("tcp://%s:%d" % (ip, port))
        if mode == 'sub':
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
        if mode == 'router' or mode == 'sub':
            self.poller.register(socket, zmq.POLLIN)
        self.logger.debug("create {} socket, at {}", mode, "tcp://%s:%d" % (ip, port))
        return socket

    def __getattr__(self, item):
        return self._sockets.get(item)

    @property
    def sockets(self):
        return list(self._sockets.keys())
