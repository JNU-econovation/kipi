#pragma checksum "D:\vscode\kipi\Clothink\Clothink\client\Shared\MainLayout.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "d7fc3f380afd391e011e0b9827436352d58a92f9"
// <auto-generated/>
#pragma warning disable 1591
namespace Clothink.Client.Shared
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
    public partial class MainLayout : LayoutComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(global::Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.OpenElement(0, "div");
            __builder.AddAttribute(1, "class", "page");
            __builder.AddAttribute(2, "b-nifs76gz0u");
            __builder.OpenElement(3, "div");
            __builder.AddAttribute(4, "class", "sidebar");
            __builder.AddAttribute(5, "style", "background-image: none; background-color: white;");
            __builder.AddAttribute(6, "b-nifs76gz0u");
            __builder.OpenComponent<global::Clothink.Client.Shared.NavMenu>(7);
            __builder.CloseComponent();
            __builder.CloseElement();
            __builder.AddMarkupContent(8, "\r\n\r\n    ");
            __builder.OpenElement(9, "div");
            __builder.AddAttribute(10, "class", "main");
            __builder.AddAttribute(11, "b-nifs76gz0u");
            __builder.AddMarkupContent(12, "<div class=\"top-row px-4\" b-nifs76gz0u><a href=\"http://blazor.net\" target=\"_blank\" class=\"ml-md-auto\" b-nifs76gz0u>About</a></div>\r\n\r\n        ");
            __builder.OpenElement(13, "div");
            __builder.AddAttribute(14, "class", "content px-4");
            __builder.AddAttribute(15, "b-nifs76gz0u");
#nullable restore
#line 14 "D:\vscode\kipi\Clothink\Clothink\client\Shared\MainLayout.razor"
__builder.AddContent(16, Body);

#line default
#line hidden
#nullable disable
            __builder.CloseElement();
            __builder.CloseElement();
            __builder.CloseElement();
        }
        #pragma warning restore 1998
    }
}
#pragma warning restore 1591
