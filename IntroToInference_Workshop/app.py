import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from flask import Flask, request, jsonify, send_file

# ---- ClimateAnomalyAnalyzer class ----
class ClimateAnomalyAnalyzer:
    def __init__(self, mu: float, sigma: float, X: float):
        self.mu = mu
        self.sigma = sigma
        self.X = X

    def compute_zscore(self) -> float:
        return (self.X - self.mu) / self.sigma

    def compute_probabilities(self) -> dict:
        Z = self.compute_zscore()
        p_leq = norm.cdf(Z)
        p_gt = 1 - p_leq
        return {"Z": round(Z, 2), "p_leq": float(p_leq), "p_gt": float(p_gt)}

    def plot_distribution(self):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 600)
        pdf = norm.pdf(x, loc=self.mu, scale=self.sigma)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, pdf, label="Normal Distribution")
        ax.axvline(self.X, color="red", linestyle="--", label=f"X = {self.X}")
        mask = x >= self.X
        ax.fill_between(x[mask], pdf[mask], 0, alpha=0.3)
        fig.tight_layout()
        return fig

# ---- Flask App ----
app = Flask(__name__)

@app.route("/analyze", methods=["GET"])
def analyze():
    mu = float(request.args.get("mu", 0.5))
    sigma = float(request.args.get("sigma", 0.2))
    X = float(request.args.get("X", 0.9))
    analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
    results = analyzer.compute_probabilities()

    if request.args.get("format", "json") == "json":
        return jsonify(results)

    fig = analyzer.plot_distribution()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
