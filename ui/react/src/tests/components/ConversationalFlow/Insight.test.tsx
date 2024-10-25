import Insight from "components/ConversationalFlow/Insight"
import renderWithProviders from "tests/utils"

test("render insight", () => {
    renderWithProviders(<Insight insightText="Hello" errorSummary="" botResponseSummary="" showErrorSummary />)
})