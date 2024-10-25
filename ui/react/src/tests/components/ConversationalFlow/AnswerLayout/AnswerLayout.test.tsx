import AnswerLayout from "components/ConversationalFlow/AnswerLayout/AnswerLayout"
import renderWithProviders from "tests/utils"

test("render the AnswerLayout", () => {
    renderWithProviders(<AnswerLayout children={<div>Hello</div>} />)
})