import prometheus_client


class REGISTRY:
    inited = False

    @classmethod
    def get_metrics(cls):
        if cls.inited:
            return cls
        else:
            # Register your metrics here
            cls.REQUEST_TIME = prometheus_client.Summary(
                "some_summary", "Time spent in processing request"
            )
            cls.inited = True
