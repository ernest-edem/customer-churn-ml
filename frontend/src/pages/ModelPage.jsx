import {
    BarChart3,
    BrainCircuit,
    CheckCircle2,
    Database,
    GitBranch,
    Layers3,
} from 'lucide-react'
import { useLocation, useNavigate } from 'react-router-dom'

const crossValidationMetrics = [
    {
        model: 'Gradient Boosting',
        accuracy: '79.96% ± 0.61%',
        precision: '79.00% ± 0.72%',
        recall: '79.96% ± 0.61%',
        f1: '79.16% ± 0.70%',
        rocAuc: '84.48% ± 0.61%',
    },
    {
        model: 'Logistic Regression',
        accuracy: '79.66% ± 0.60%',
        precision: '78.89% ± 0.67%',
        recall: '79.66% ± 0.60%',
        f1: '79.12% ± 0.65%',
        rocAuc: '84.30% ± 0.52%',
    },
    {
        model: 'Random Forest',
        accuracy: '77.81% ± 0.61%',
        precision: '76.73% ± 0.60%',
        recall: '77.81% ± 0.61%',
        f1: '77.05% ± 0.58%',
        rocAuc: '81.28% ± 0.50%',
    },
    {
        model: 'Decision Tree',
        accuracy: '73.55% ± 1.12%',
        precision: '73.68% ± 0.94%',
        recall: '73.55% ± 1.12%',
        f1: '73.61% ± 1.02%',
        rocAuc: '66.83% ± 1.28%',
    },
]

const modelBenchmark = [
    {
        model: 'Gradient Boosting',
        accuracy: '78.75%',
        precision: '77.81%',
        recall: '78.75%',
        f1: '78.11%',
        rocAuc: '83.44%',
    },
    {
        model: 'Logistic Regression',
        accuracy: '78.82%',
        precision: '78.06%',
        recall: '78.82%',
        f1: '78.33%',
        rocAuc: '83.39%',
    },
    {
        model: 'Random Forest',
        accuracy: '77.33%',
        precision: '76.24%',
        recall: '77.33%',
        f1: '76.59%',
        rocAuc: '79.23%',
    },
    {
        model: 'Decision Tree',
        accuracy: '71.29%',
        precision: '71.63%',
        recall: '71.29%',
        f1: '71.45%',
        rocAuc: '64.20%',
    },
]

function MetricCard({ label, value }) {
    return (
        <div className="rounded-xl border border-border bg-surface p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                {label}
            </p>
            <p className="mt-2 text-2xl font-bold tracking-tight text-text">
                {value}
            </p>
        </div>
    )
}

function InfoCard({ icon: Icon, title, children }) {
    return (
        <div className="rounded-xl border border-border bg-surface p-5">
            <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-soft text-primary">
                    <Icon size={18} />
                </div>
                <h3 className="font-semibold text-text">{title}</h3>
            </div>
            <div className="mt-4 text-sm leading-6 text-text-muted">
                {children}
            </div>
        </div>
    )
}

function ModelPage() {
    const location = useLocation()
    const navigate = useNavigate()

    const assessment = location.state?.assessment
    const prediction = location.state?.prediction

    const resultState =
        assessment && prediction
            ? {
                assessment,
                prediction,
            }
            : undefined

    return (
        <div className="min-h-screen bg-background text-text">
            <header className="border-b border-border bg-surface">
                <div className="mx-auto flex max-w-7xl flex-col gap-5 px-6 py-5 lg:px-8 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-white">
                            <BrainCircuit size={22} strokeWidth={2} />
                        </div>
                        <div>
                            <h1 className="text-lg font-semibold tracking-tight">
                                Customer Churn ML
                            </h1>
                            <p className="text-sm text-text-muted">
                                Model information
                            </p>
                        </div>
                    </div>

                    <nav
                        aria-label="Model page navigation"
                        className="flex flex-wrap items-center gap-2"
                    >
                        <button
                            type="button"
                            onClick={() => navigate('/')}
                            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Dashboard
                        </button>

                        <button
                            type="button"
                            onClick={() =>
                                navigate('/results', {
                                    state: resultState,
                                })
                            }
                            className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Results
                        </button>

                        <button
                            type="button"
                            onClick={() => navigate('/assessment')}
                            className="rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            New Assessment
                        </button>
                    </nav>
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
                <div className="mb-8">
                    <p className="text-sm font-semibold text-primary">
                        Machine Learning Model
                    </p>
                    <h2 className="mt-1 text-3xl font-bold tracking-tight text-text">
                        Model Performance
                    </h2>
                    <p className="mt-2 max-w-3xl text-sm leading-6 text-text-muted">
                        Evaluation results for the customer churn prediction system,
                        including holdout performance and five-fold cross-validation.
                    </p>
                </div>

                <section>
                    <div className="mb-4 flex items-center gap-2">
                        <BarChart3 size={19} className="text-primary" />
                        <h3 className="text-lg font-semibold text-text">
                            Holdout Test Performance
                        </h3>
                    </div>

                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                        <MetricCard label="Accuracy" value="78.75%" />
                        <MetricCard label="Precision" value="77.81%" />
                        <MetricCard label="Recall" value="78.75%" />
                        <MetricCard label="F1 Score" value="78.11%" />
                        <MetricCard label="ROC-AUC" value="83.44%" />
                    </div>
                </section>

                <section className="mt-10">
                    <div className="mb-4">
                        <h3 className="text-lg font-semibold text-text">
                            5-Fold Cross-Validation
                        </h3>
                        <p className="mt-1 text-sm text-text-muted">
                            Mean ± standard deviation across stratified folds on the
                            training split.
                        </p>
                    </div>

                    <div className="overflow-x-auto rounded-xl border border-border bg-surface">
                        <table className="min-w-full text-left text-sm">
                            <thead className="border-b border-border bg-surface-muted">
                                <tr>
                                    {[
                                        'Model',
                                        'Accuracy',
                                        'Precision',
                                        'Recall',
                                        'F1',
                                        'ROC-AUC',
                                    ].map((heading) => (
                                        <th
                                            key={heading}
                                            className="whitespace-nowrap px-4 py-3 font-semibold text-text"
                                        >
                                            {heading}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {crossValidationMetrics.map((row) => (
                                    <tr
                                        key={row.model}
                                        className="border-b border-border last:border-b-0"
                                    >
                                        <td className="whitespace-nowrap px-4 py-3 font-medium text-text">
                                            {row.model}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.accuracy}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.precision}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.recall}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.f1}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.rocAuc}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                <section className="mt-10">
                    <div className="mb-4">
                        <h3 className="text-lg font-semibold text-text">
                            Holdout Model Comparison
                        </h3>
                        <p className="mt-1 text-sm text-text-muted">
                            Comparison of the four evaluated classification models on the
                            reserved test set.
                        </p>
                    </div>

                    <div className="overflow-x-auto rounded-xl border border-border bg-surface">
                        <table className="min-w-full text-left text-sm">
                            <thead className="border-b border-border bg-surface-muted">
                                <tr>
                                    {[
                                        'Model',
                                        'Accuracy',
                                        'Precision',
                                        'Recall',
                                        'F1',
                                        'ROC-AUC',
                                    ].map((heading) => (
                                        <th
                                            key={heading}
                                            className="whitespace-nowrap px-4 py-3 font-semibold text-text"
                                        >
                                            {heading}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {modelBenchmark.map((row) => (
                                    <tr
                                        key={row.model}
                                        className="border-b border-border last:border-b-0"
                                    >
                                        <td className="whitespace-nowrap px-4 py-3 font-medium text-text">
                                            {row.model}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.accuracy}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.precision}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.recall}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.f1}
                                        </td>
                                        <td className="whitespace-nowrap px-4 py-3 text-text-muted">
                                            {row.rocAuc}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                <section className="mt-10 grid gap-4 md:grid-cols-2">
                    <InfoCard icon={CheckCircle2} title="Selected Model">
                        <p>
                            Gradient Boosting is the persisted model used by the prediction
                            API. It was selected based on the holdout model comparison.
                        </p>
                    </InfoCard>

                    <InfoCard icon={GitBranch} title="Evaluation Strategy">
                        <p>
                            An 80/20 stratified split is used for the final holdout
                            evaluation. Five-fold stratified cross-validation is applied only
                            to the training split.
                        </p>
                    </InfoCard>

                    <InfoCard icon={Database} title="Data & Features">
                        <p>
                            The processed dataset contains 7,032 customer records. Mutual
                            Information SelectKBest selects the top 10 predictive features.
                        </p>
                    </InfoCard>

                    <InfoCard icon={Layers3} title="Preprocessing">
                        <p>
                            Numerical features use imputation and standard scaling.
                            Categorical features use imputation and one-hot encoding with
                            unknown categories ignored.
                        </p>
                    </InfoCard>
                </section>

                <div className="mt-8 rounded-xl border border-border bg-surface-muted p-4 text-sm leading-6 text-text-muted">
                    These metrics describe performance on the available dataset and
                    evaluation splits. They do not guarantee performance on unseen
                    real-world customers.
                </div>
            </main>
        </div>
    )
}

export default ModelPage