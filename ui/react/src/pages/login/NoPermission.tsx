import InsightsPro from '../../assets/images/Insightsprologo.png'

export default function NoPermission() {
  const goToLogin = () => {
    window.location.href = window.location.origin
  }
  return (
    <div className="w-[100%] h-[100vh] bg-gray-100 flex flex-col">
      <div className="h-[92%] w-[99%] flex justify-center items-center align-center">
        <div className="flex  py-5 px-20 flex-col justify-between rounded-[20px] min-h-[300px] min-w-[400px] bg-white">
          <div className="flex mt-10 justify-center text-lg"><img src={InsightsPro} alt="Insights Pro" /></div>
          <div className=" justify-center align-end content-center">
            <button data-testid="routechange-button" type="button"
              className="w-full bg-[#18749C] text-[14px] font-poppins text-white font-medium py-2 flex justify-center content-center rounded"
              onClick={() => goToLogin()}
            >
              Login
            </button>
          </div>
          <div className="text-xs text-red-500">
            You do not have permission to access the application
          </div>
          <div className="flex justify-center text-[12px] text-[#9B9898]">Powered by Tiger Analytics </div>
        </div>
      </div>
    </div>
  )
}