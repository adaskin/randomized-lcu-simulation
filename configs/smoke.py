"""Tiny config (seconds) to check the pipeline end to end; also a template for new experiments."""
config = dict(
    name="smoke",
    seed=0,
    base=dict(tau=20.0, nt=5, ms=[2, 4, 8], nrand=1),
    cases=[
        dict(tag="A n=6", hamiltonian=dict(model="A", n=6), drift_k_frac=0.1),
        dict(tag="B n=6 L=40", hamiltonian=dict(model="B", n=6, L=40, seed=0)),
        dict(tag="val A n=6", task="validate", tau=4.0, nt=5, hamiltonian=dict(model="A", n=6),
             validate=dict(plans=[dict(method="qDRIFT"), dict(method="graph", k_div=4)], r=[50], ntraj=20)),
    ],
)
