import InsightNumber from "components/ConversationalFlow/InsightNumber/InsightNumber"
import renderWithProviders from "tests/utils"

test("render the InsightNumber", () => {
    renderWithProviders(<InsightNumber num="33" />)
})