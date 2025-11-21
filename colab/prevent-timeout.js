const startClickConnect = () => {
  const clickConnect = () => {
    console.log("Connnect Clicked - Start");
    document
      .querySelector("#top-toolbar > colab-connect-button")
      .shadowRoot.querySelector("#connect")
      .click();
    console.log("Connnect Clicked - End");
  };

  const intervalId = setInterval(clickConnect, 60000);

  const stopClickConnectHandler = () => {
    console.log("Connnect Clicked Stopped - Start");
    clearInterval(intervalId);
    console.log("Connnect Clicked Stopped - End");
  };

  return stopClickConnectHandler;
};

const stopClickConnect = startClickConnect();
