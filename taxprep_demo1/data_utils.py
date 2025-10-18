import pandas as pd
import numpy as np

def load_or_generate_sample(n=200):
    np.random.seed(42)
    client_ids = [f'C{10000+i}' for i in range(n)]
    df = pd.DataFrame(
        {
            "client_id": client_ids,
            "service_type": np.random.choice(
                ["Individual", "Business", "Consulting"], size=n
            ),
            "turnaround_time_days": np.random.poisson(7, size=n)
            + np.random.randint(0, 10, size=n),
            "error_rate_pct": np.round(np.random.beta(1.5, 30, size=n) * 100, 2),
            "advisor_experience_years": np.round(np.random.exponential(3, size=n), 2),
            "communication_count": np.random.randint(0, 8, size=n),
            "unresolved_tickets": np.random.randint(0, 3, size=n),
            "last_feedback_text": np.random.choice(
                [
                    "Great service",
                    "Late delivery",
                    "Incorrect forms",
                    "Helpful advisor",
                ],
                size=n,
            ),
        }
    )
    return df
