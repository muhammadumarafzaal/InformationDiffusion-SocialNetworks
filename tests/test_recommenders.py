"""Behavioural tests for the hand-written recommenders.

Each assertion encodes a property the report relies on: the bias model beats the
global mean, the latent-factor model beats the bias model on the training fold,
the social regulariser is a real knob (beta = 0 is an exact control), and every
model is deterministic for a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sntp.data import TRAIN
from sntp.metrics import rmse
from sntp.recommenders import (
    BiasBaseline,
    BiasMF,
    GlobalMeanRecommender,
    ItemKNN,
    PopularityRecommender,
    SocialMF,
    build_recommender,
)
from sntp.recommenders.social import NeighbourResidualRecommender


def _train_error(model, dataset) -> float:
    users, items, truth = dataset.interactions((TRAIN,))
    return rmse(model.predict(users, items), truth)


def test_global_mean_predicts_a_constant(dataset):
    model = GlobalMeanRecommender().fit(dataset)
    matrix = model.predict_matrix(np.arange(5))
    assert np.allclose(matrix, dataset.ratings[dataset.fold == TRAIN].mean())


def test_bias_baseline_beats_the_global_mean(dataset):
    global_mean = GlobalMeanRecommender().fit(dataset)
    bias = BiasBaseline(reg=5.0).fit(dataset)
    assert _train_error(bias, dataset) < _train_error(global_mean, dataset)


def test_popularity_is_non_personalised(dataset):
    popularity = PopularityRecommender().fit(dataset)
    matrix = popularity.predict_matrix(np.arange(10))
    # every user gets the same ranking, which is what makes it a ranking baseline
    assert np.allclose(matrix, matrix[0])


def test_bias_mf_improves_the_training_error_and_is_deterministic(dataset, cfg):
    bias = BiasBaseline(reg=10.0).fit(dataset)
    first = build_recommender("bias_mf", cfg, seed=11).fit(dataset)
    second = build_recommender("bias_mf", cfg, seed=11).fit(dataset)
    assert _train_error(first, dataset) < _train_error(bias, dataset)
    assert np.allclose(first.predict_matrix(np.arange(10)), second.predict_matrix(np.arange(10)))
    assert first.history["train_rmse"][-1] < first.history["train_rmse"][0]


def test_bias_mf_predict_agrees_with_predict_matrix(dataset, cfg):
    model = build_recommender("bias_mf", cfg, seed=3).fit(dataset)
    users = np.array([0, 2, 2])
    items = np.array([1, 5, 7])
    matrix = model.predict_matrix(np.arange(dataset.n_users))
    assert np.allclose(model.predict(users, items), matrix[users, items])


def test_item_knn_beats_the_global_mean(dataset):
    global_mean = GlobalMeanRecommender().fit(dataset)
    knn = ItemKNN(n_neighbors=10, shrinkage=2.0).fit(dataset)
    assert _train_error(knn, dataset) < _train_error(global_mean, dataset)
    assert np.isfinite(knn.predict_matrix(np.arange(4))).all()


def test_social_mf_with_beta_zero_reproduces_bias_mf(dataset, cfg):
    graph = dataset.influence_graph()
    plain = build_recommender("bias_mf", cfg, seed=5).fit(dataset)
    social = SocialMF(n_factors=8, epochs=6, batch_size=256, beta=0.0, seed=5).fit(
        dataset, graph=graph
    )
    assert social.history["social_norm"] == []
    assert np.allclose(
        plain.predict_matrix(np.arange(10)), social.predict_matrix(np.arange(10)), atol=1e-6
    )


def test_social_regulariser_changes_the_solution(dataset):
    graph = dataset.influence_graph()
    plain = SocialMF(n_factors=8, epochs=6, batch_size=256, beta=0.0, seed=5).fit(
        dataset, graph=graph
    )
    social = SocialMF(n_factors=8, epochs=6, batch_size=256, beta=1.0, seed=5).fit(
        dataset, graph=graph
    )
    assert social.history["social_norm"]
    assert not np.allclose(
        plain.predict_matrix(np.arange(10)), social.predict_matrix(np.arange(10))
    )


def test_neighbour_residual_degrades_to_the_bias_model_without_edges(dataset):
    from scipy import sparse

    bias = BiasBaseline(reg=10.0).fit(dataset)
    empty = NeighbourResidualRecommender(sparse.csr_matrix((dataset.n_users,) * 2), "empty")
    empty.fit(dataset)
    assert empty.coverage == 0.0
    assert np.allclose(
        empty.predict_matrix(np.arange(10)), bias.predict_matrix(np.arange(10)), atol=1e-6
    )

    social = NeighbourResidualRecommender(dataset.influence_graph(), "trust").fit(dataset)
    assert 0.0 < social.coverage <= 1.0
    assert np.isfinite(social.predict_matrix(np.arange(10))).all()
    # the neighbour signal must actually change the prediction somewhere
    assert not np.allclose(social.predict_matrix(np.arange(10)), bias.predict_matrix(np.arange(10)))


def test_factory_rejects_unknown_models(cfg):
    with pytest.raises(ValueError):
        build_recommender("no_such_model", cfg)


def test_unfitted_models_refuse_to_predict():
    with pytest.raises(RuntimeError):
        BiasMF(n_factors=4).predict_matrix(np.array([0]))
