import { useState } from 'react'
import {
    Check,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    CreditCard,
    MonitorPlay,
    Phone,
    RotateCcw,
    ShieldCheck,
    UserRound,
    Users,
} from 'lucide-react'

const initialFormData = {
    gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 0,
    PhoneService: 'Yes',
    MultipleLines: 'No phone service',
    InternetService: 'DSL',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 0,
    TotalCharges: 0,
}

const selectOptions = {
    gender: ['Female', 'Male'],
    SeniorCitizen: ['0', '1'],
    Partner: ['Yes', 'No'],
    Dependents: ['Yes', 'No'],
    PhoneService: ['Yes', 'No'],
    MultipleLines: ['No phone service', 'No', 'Yes'],
    InternetService: ['DSL', 'Fiber optic', 'No'],
    OnlineSecurity: ['No', 'Yes', 'No internet service'],
    OnlineBackup: ['No', 'Yes', 'No internet service'],
    DeviceProtection: ['No', 'Yes', 'No internet service'],
    TechSupport: ['No', 'Yes', 'No internet service'],
    StreamingTV: ['No', 'Yes', 'No internet service'],
    StreamingMovies: ['No', 'Yes', 'No internet service'],
    Contract: ['Month-to-month', 'One year', 'Two year'],
    PaperlessBilling: ['Yes', 'No'],
    PaymentMethod: [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
    ],
}

const steps = [
    {
        id: 'profile',
        title: 'Customer Profile',
        description: 'Demographic and household information',
    },
    {
        id: 'services',
        title: 'Services',
        description: 'Phone, internet and support services',
    },
    {
        id: 'billing',
        title: 'Billing & Usage',
        description: 'Streaming, contract and billing details',
    },
    {
        id: 'review',
        title: 'Review',
        description: 'Confirm information before prediction',
    },
]

function FieldLabel({ children }) {
    return (
        <label className="mb-2 block text-sm font-medium text-text">
            {children}
            <span className="ml-1 text-danger">*</span>
        </label>
    )
}

function SelectField({ label, name, value, options, onChange }) {
    return (
        <div>
            <FieldLabel>{label}</FieldLabel>

            <div className="relative">
                <select
                    name={name}
                    value={value}
                    onChange={onChange}
                    className="w-full appearance-none rounded-lg border border-border bg-surface px-3.5 py-2.5 pr-10 text-sm text-text outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20"
                >
                    {options.map((option) => (
                        <option key={option} value={option}>
                            {option}
                        </option>
                    ))}
                </select>

                <ChevronDown
                    size={17}
                    className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-text-muted"
                />
            </div>
        </div>
    )
}

function NumberField({
    label,
    name,
    value,
    min,
    max,
    step = 1,
    onChange,
    error,
}) {
    return (
        <div>
            <FieldLabel>{label}</FieldLabel>

            <input
                type="number"
                name={name}
                value={value}
                min={min}
                max={max}
                step={step}
                onChange={onChange}
                className={`w-full rounded-lg border bg-surface px-3.5 py-2.5 text-sm text-text outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20 ${error ? 'border-danger' : 'border-border'
                    }`}
            />

            <p className="mt-1.5 text-xs text-text-muted">
                Range: {min} to {max}
            </p>

            {error && (
                <p className="mt-1.5 text-xs font-medium text-danger">{error}</p>
            )}
        </div>
    )
}

function SectionHeader({ icon: Icon, title, description }) {
    return (
        <div className="mb-6 flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary-soft text-primary">
                <Icon size={20} />
            </div>

            <div>
                <h2 className="text-base font-semibold text-text">{title}</h2>
                <p className="mt-0.5 text-sm text-text-muted">{description}</p>
            </div>
        </div>
    )
}

function ReviewRow({ label, value }) {
    return (
        <div className="flex items-center justify-between gap-4 border-b border-border py-3 last:border-b-0">
            <span className="text-sm text-text-muted">{label}</span>
            <span className="text-right text-sm font-medium text-text">{value}</span>
        </div>
    )
}

function CustomerAssessmentForm({ onSubmit, onCancel }) {
    const [formData, setFormData] = useState(initialFormData)
    const [currentStep, setCurrentStep] = useState(0)
    const [errors, setErrors] = useState({})

    const handleChange = (event) => {
        const { name, value, type } = event.target

        setFormData((current) => ({
            ...current,
            [name]:
                type === 'number'
                    ? value === ''
                        ? ''
                        : Number(value)
                    : value,
        }))

        setErrors((current) => {
            if (!current[name]) {
                return current
            }

            const nextErrors = { ...current }
            delete nextErrors[name]
            return nextErrors
        })
    }

    const validateStep = () => {
        const nextErrors = {}

        if (currentStep === 2) {
            const numericFields = [
                ['tenure', 0, 100],
                ['MonthlyCharges', 0, 1000],
                ['TotalCharges', 0, 10000],
            ]

            numericFields.forEach(([name, min, max]) => {
                const value = formData[name]

                if (value === '' || value === null || value === undefined) {
                    nextErrors[name] = 'This field is required.'
                    return
                }

                if (Number.isNaN(Number(value)) || value < min || value > max) {
                    nextErrors[name] = `Enter a value between ${min} and ${max}.`
                }
            })
        }

        setErrors(nextErrors)

        return Object.keys(nextErrors).length === 0
    }

    const handleNext = () => {
        if (!validateStep()) {
            return
        }

        setCurrentStep((current) => Math.min(current + 1, steps.length - 1))
    }

    const handlePrevious = () => {
        setErrors({})
        setCurrentStep((current) => Math.max(current - 1, 0))
    }

    const handleReset = () => {
        setFormData(initialFormData)
        setErrors({})
        setCurrentStep(0)
    }

    const handleSubmit = (event) => {
        event.preventDefault()

        if (!validateStep()) {
            return
        }

        const payload = {
            ...formData,
            tenure: Number(formData.tenure),
            MonthlyCharges: Number(formData.MonthlyCharges),
            TotalCharges: Number(formData.TotalCharges),
            SeniorCitizen: Number(formData.SeniorCitizen),
        }

        onSubmit?.(payload)
    }

    const isReviewStep = currentStep === steps.length - 1
    const progress = ((currentStep + 1) / steps.length) * 100

    return (
        <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
            <div className="mb-8">
                <div className="mb-6">
                    <p className="text-sm font-medium text-primary">
                        Customer Assessment
                    </p>

                    <h1 className="mt-1 text-2xl font-bold tracking-tight text-text sm:text-3xl">
                        Assess customer churn risk
                    </h1>

                    <p className="mt-2 max-w-2xl text-sm leading-6 text-text-muted">
                        Provide the customer information used by the trained machine
                        learning pipeline to estimate churn probability.
                    </p>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-surface-muted">
                    <div
                        className="h-full rounded-full bg-primary transition-all duration-300"
                        style={{ width: `${progress}%` }}
                    />
                </div>

                <div className="mt-5 grid grid-cols-4 gap-2">
                    {steps.map((step, index) => {
                        const isActive = index === currentStep
                        const isComplete = index < currentStep

                        return (
                            <div key={step.id} className="min-w-0">
                                <div className="flex items-center gap-2">
                                    <div
                                        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${isComplete
                                            ? 'bg-primary text-white'
                                            : isActive
                                                ? 'bg-primary text-white'
                                                : 'bg-surface-muted text-text-muted'
                                            }`}
                                    >
                                        {isComplete ? <Check size={15} /> : index + 1}
                                    </div>

                                    <span
                                        className={`hidden truncate text-sm font-medium sm:block ${isActive ? 'text-text' : 'text-text-muted'
                                            }`}
                                    >
                                        {step.title}
                                    </span>
                                </div>

                                {index < steps.length - 1 && (
                                    <div className="mt-2 hidden h-px bg-border sm:block" />
                                )}
                            </div>
                        )
                    })}
                </div>
            </div>

            <section className="rounded-2xl border border-border bg-surface p-6 shadow-sm sm:p-8">
                {currentStep === 0 && (
                    <>
                        <SectionHeader
                            icon={UserRound}
                            title={steps[0].title}
                            description={steps[0].description}
                        />

                        <div className="grid gap-5 md:grid-cols-2">
                            <SelectField
                                label="Gender"
                                name="gender"
                                value={formData.gender}
                                options={selectOptions.gender}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Senior citizen"
                                name="SeniorCitizen"
                                value={String(formData.SeniorCitizen)}
                                options={selectOptions.SeniorCitizen}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Partner"
                                name="Partner"
                                value={formData.Partner}
                                options={selectOptions.Partner}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Dependents"
                                name="Dependents"
                                value={formData.Dependents}
                                options={selectOptions.Dependents}
                                onChange={handleChange}
                            />

                            <NumberField
                                label="Tenure (months)"
                                name="tenure"
                                value={formData.tenure}
                                min={0}
                                max={100}
                                onChange={handleChange}
                                error={errors.tenure}
                            />
                        </div>
                    </>
                )}

                {currentStep === 1 && (
                    <>
                        <SectionHeader
                            icon={Phone}
                            title={steps[1].title}
                            description={steps[1].description}
                        />

                        <div className="grid gap-5 md:grid-cols-2">
                            <SelectField
                                label="Phone service"
                                name="PhoneService"
                                value={formData.PhoneService}
                                options={selectOptions.PhoneService}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Multiple lines"
                                name="MultipleLines"
                                value={formData.MultipleLines}
                                options={selectOptions.MultipleLines}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Internet service"
                                name="InternetService"
                                value={formData.InternetService}
                                options={selectOptions.InternetService}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Online security"
                                name="OnlineSecurity"
                                value={formData.OnlineSecurity}
                                options={selectOptions.OnlineSecurity}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Online backup"
                                name="OnlineBackup"
                                value={formData.OnlineBackup}
                                options={selectOptions.OnlineBackup}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Device protection"
                                name="DeviceProtection"
                                value={formData.DeviceProtection}
                                options={selectOptions.DeviceProtection}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Technical support"
                                name="TechSupport"
                                value={formData.TechSupport}
                                options={selectOptions.TechSupport}
                                onChange={handleChange}
                            />
                        </div>
                    </>
                )}

                {currentStep === 2 && (
                    <>
                        <SectionHeader
                            icon={CreditCard}
                            title={steps[2].title}
                            description={steps[2].description}
                        />

                        <div className="grid gap-5 md:grid-cols-2">
                            <SelectField
                                label="Streaming TV"
                                name="StreamingTV"
                                value={formData.StreamingTV}
                                options={selectOptions.StreamingTV}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Streaming movies"
                                name="StreamingMovies"
                                value={formData.StreamingMovies}
                                options={selectOptions.StreamingMovies}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Contract"
                                name="Contract"
                                value={formData.Contract}
                                options={selectOptions.Contract}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Paperless billing"
                                name="PaperlessBilling"
                                value={formData.PaperlessBilling}
                                options={selectOptions.PaperlessBilling}
                                onChange={handleChange}
                            />

                            <SelectField
                                label="Payment method"
                                name="PaymentMethod"
                                value={formData.PaymentMethod}
                                options={selectOptions.PaymentMethod}
                                onChange={handleChange}
                            />

                            <NumberField
                                label="Monthly charges"
                                name="MonthlyCharges"
                                value={formData.MonthlyCharges}
                                min={0}
                                max={1000}
                                step={0.01}
                                onChange={handleChange}
                                error={errors.MonthlyCharges}
                            />

                            <NumberField
                                label="Total charges"
                                name="TotalCharges"
                                value={formData.TotalCharges}
                                min={0}
                                max={10000}
                                step={0.01}
                                onChange={handleChange}
                                error={errors.TotalCharges}
                            />
                        </div>
                    </>
                )}

                {currentStep === 3 && (
                    <>
                        <SectionHeader
                            icon={ShieldCheck}
                            title={steps[3].title}
                            description={steps[3].description}
                        />

                        <div className="grid gap-6 lg:grid-cols-2">
                            <div className="rounded-xl border border-border bg-background p-5">
                                <div className="mb-3 flex items-center gap-2">
                                    <Users size={18} className="text-primary" />
                                    <h3 className="text-sm font-semibold text-text">
                                        Customer Profile
                                    </h3>
                                </div>

                                <ReviewRow label="Gender" value={formData.gender} />
                                <ReviewRow
                                    label="Senior citizen"
                                    value={formData.SeniorCitizen}
                                />
                                <ReviewRow label="Partner" value={formData.Partner} />
                                <ReviewRow label="Dependents" value={formData.Dependents} />
                                <ReviewRow label="Tenure" value={`${formData.tenure} months`} />
                            </div>

                            <div className="rounded-xl border border-border bg-background p-5">
                                <div className="mb-3 flex items-center gap-2">
                                    <Phone size={18} className="text-primary" />
                                    <h3 className="text-sm font-semibold text-text">
                                        Services
                                    </h3>
                                </div>

                                <ReviewRow
                                    label="Phone service"
                                    value={formData.PhoneService}
                                />
                                <ReviewRow
                                    label="Multiple lines"
                                    value={formData.MultipleLines}
                                />
                                <ReviewRow
                                    label="Internet service"
                                    value={formData.InternetService}
                                />
                                <ReviewRow
                                    label="Online security"
                                    value={formData.OnlineSecurity}
                                />
                                <ReviewRow
                                    label="Online backup"
                                    value={formData.OnlineBackup}
                                />
                                <ReviewRow
                                    label="Device protection"
                                    value={formData.DeviceProtection}
                                />
                                <ReviewRow
                                    label="Technical support"
                                    value={formData.TechSupport}
                                />
                            </div>

                            <div className="rounded-xl border border-border bg-background p-5">
                                <div className="mb-3 flex items-center gap-2">
                                    <MonitorPlay size={18} className="text-primary" />
                                    <h3 className="text-sm font-semibold text-text">
                                        Streaming
                                    </h3>
                                </div>

                                <ReviewRow
                                    label="Streaming TV"
                                    value={formData.StreamingTV}
                                />
                                <ReviewRow
                                    label="Streaming movies"
                                    value={formData.StreamingMovies}
                                />
                            </div>

                            <div className="rounded-xl border border-border bg-background p-5">
                                <div className="mb-3 flex items-center gap-2">
                                    <CreditCard size={18} className="text-primary" />
                                    <h3 className="text-sm font-semibold text-text">
                                        Billing
                                    </h3>
                                </div>

                                <ReviewRow label="Contract" value={formData.Contract} />
                                <ReviewRow
                                    label="Paperless billing"
                                    value={formData.PaperlessBilling}
                                />
                                <ReviewRow
                                    label="Payment method"
                                    value={formData.PaymentMethod}
                                />
                                <ReviewRow
                                    label="Monthly charges"
                                    value={formData.MonthlyCharges}
                                />
                                <ReviewRow
                                    label="Total charges"
                                    value={formData.TotalCharges}
                                />
                            </div>
                        </div>

                        <div className="mt-6 rounded-xl border border-primary/20 bg-primary-soft/50 p-5">
                            <p className="text-sm leading-6 text-text-muted">
                                Review the information above before submitting. The final
                                prediction will be generated using the trained Gradient
                                Boosting model through the FastAPI prediction service.
                            </p>
                        </div>
                    </>
                )}
            </section>

            <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex gap-3">
                    {onCancel && (
                        <button
                            type="button"
                            onClick={onCancel}
                            className="rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Cancel
                        </button>
                    )}

                    <button
                        type="button"
                        onClick={handleReset}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                        <RotateCcw size={16} />
                        Reset
                    </button>
                </div>

                <div className="flex gap-3">
                    {currentStep > 0 && (
                        <button
                            type="button"
                            onClick={handlePrevious}
                            className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface px-5 py-2.5 text-sm font-semibold text-text transition hover:bg-surface-muted focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            <ChevronLeft size={17} />
                            Previous
                        </button>
                    )}

                    {!isReviewStep ? (
                        <button
                            type="button"
                            onClick={handleNext}
                            className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Next
                            <ChevronRight size={17} />
                        </button>
                    ) : (
                        <button
                            type="submit"
                            className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-primary-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            Predict Churn
                            <ChevronRight size={17} />
                        </button>
                    )}
                </div>
            </div>
        </form>
    )
}

export default CustomerAssessmentForm