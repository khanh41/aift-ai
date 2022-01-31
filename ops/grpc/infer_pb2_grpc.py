# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from ops.grpc import infer_pb2 as ops_dot_grpc_dot_infer__pb2


class ExercisePredictStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Inference = channel.unary_unary(
                '/ExercisePredict/Inference',
                request_serializer=ops_dot_grpc_dot_infer__pb2.ExercisePredictRequest.SerializeToString,
                response_deserializer=ops_dot_grpc_dot_infer__pb2.ImgBase64.FromString,
                )


class ExercisePredictServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Inference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExercisePredictServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Inference': grpc.unary_unary_rpc_method_handler(
                    servicer.Inference,
                    request_deserializer=ops_dot_grpc_dot_infer__pb2.ExercisePredictRequest.FromString,
                    response_serializer=ops_dot_grpc_dot_infer__pb2.ImgBase64.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ExercisePredict', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ExercisePredict(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExercisePredict/Inference',
            ops_dot_grpc_dot_infer__pb2.ExercisePredictRequest.SerializeToString,
            ops_dot_grpc_dot_infer__pb2.ImgBase64.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
