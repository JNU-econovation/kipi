#pragma checksum "D:\vscode\kipi\Clothink\Clothink\client\Pages\Home.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "a87cf41cbd9a931abdd8dea383e3a5f566285916"
// <auto-generated/>
#pragma warning disable 1591
namespace Clothink.Client.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using System.Net.Http.Json;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.AspNetCore.Components.Web.Virtualization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.AspNetCore.Components.WebAssembly.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Clothink.Client;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "D:\vscode\kipi\Clothink\Clothink\client\_Imports.razor"
using Clothink.Client.Shared;

#line default
#line hidden
#nullable disable
    [global::Microsoft.AspNetCore.Components.RouteAttribute("/")]
    public partial class Home : global::Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.AddMarkupContent(0, @"<html><head><title>오늘의 날씨</title>
    <link rel=""stylesheet"" href=""styles.css""></head>
<body><div class=""weather-card""><h2 class=""location"">오늘의 날씨 <span class=""cur-location""></span></h2>
        <p class=""temperature"">온도: <span class=""cur-temp""></span>°C</p>
        <p class=""clouds"">구름 상태: <span class=""cur-clouds""></span>%</p></div></body></html>");
        }
        #pragma warning restore 1998
    }
}
#pragma warning restore 1591
