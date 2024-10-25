import QuestionLayout from "components/ConversationalFlow/QuestionLayout/QuestionLayout"
import renderWithProviders from "tests/utils"

test("render QuestionLayout", () => {
    renderWithProviders(<QuestionLayout children={<div>Hello</div>} />)
})